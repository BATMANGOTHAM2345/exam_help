import os
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cloudinary
import cloudinary.uploader
import fitz
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pymongo import MongoClient

load_dotenv()

db = MongoClient(os.getenv("MONGODB_URL"))["EXAMHELP_APP"]
vision_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    api_key=os.getenv("GROQ_API_KEY"),
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

FAISS_DIR = Path("faiss")
FAISS_DIR.mkdir(parents=True, exist_ok=True)
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash-lite")


def vector_store_path(pdf_id):
    return FAISS_DIR / pdf_id


def extract_normal_pdf_pages(pdf_path):
    return [doc.page_content.strip() for doc in PyPDFLoader(str(pdf_path)).load()]


def vision_extract_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    response = vision_client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            "Extract all readable text from this page. Keep headings, bullets, paragraph text, "
            "diagram labels, and captions. Do not summarize or explain. Return only extracted text.",
            genai.types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
        ],
    )
    return (response.text or "").strip()


def extract_scanned_pdf_pages(pdf_path):
    pages = []
    with fitz.open(str(pdf_path)) as pdf:
        for page in pdf:
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png")))
            pages.append(vision_extract_image(image))
    return pages


def extract_pdf_pages_from_file(pdf_path, extraction_mode):
    return extract_scanned_pdf_pages(pdf_path) if extraction_mode == "scanned" else extract_normal_pdf_pages(pdf_path)


def build_vector_store(pdf_id, pdf_bytes, extraction_mode):
    temp = Path(tempfile.gettempdir()) / f"{pdf_id}.pdf"
    try:
        temp.write_bytes(bytes(pdf_bytes))
        texts = []
        for page in extract_pdf_pages_from_file(temp, extraction_mode):
            texts.extend(splitter.split_text(page))
        texts = [text for text in texts if text.strip()]

        if not texts:
            message = "No readable text found. Make sure the scan is clear."
            if extraction_mode == "normal":
                message = "No readable text found. Upload this file as a scanned PDF."
            raise RuntimeError(message)

        path = vector_store_path(pdf_id)
        path.mkdir(parents=True, exist_ok=True)
        FAISS.from_texts(texts, embeddings).save_local(str(path))
        return path
    finally:
        if temp.exists():
            temp.unlink()


def ensure_vector_store(pdf):
    path = Path(pdf.get("vector_path") or vector_store_path(pdf["pdf_id"]))
    if (path / "index.faiss").exists() and (path / "index.pkl").exists():
        return path

    rebuilt_path = build_vector_store(
        pdf["pdf_id"],
        fetch_pdf_bytes(pdf["pdf_url"]),
        pdf.get("extraction_mode", "normal"),
    )
    db.pdfs.update_one({"pdf_id": pdf["pdf_id"]}, {"$set": {"vector_path": str(rebuilt_path)}})
    return rebuilt_path


def upload_pdf(file, folder_id, extraction_mode):
    pdf_id = str(uuid4())
    pdf_bytes = bytes(file.getbuffer())
    temp = Path(tempfile.gettempdir()) / f"{pdf_id}.pdf"
    temp.write_bytes(pdf_bytes)

    try:
        url = cloudinary.uploader.upload(
            str(temp),
            resource_type="raw",
            folder=os.getenv("CLOUDINARY_UPLOAD_FOLDER"),
            public_id=pdf_id,
            overwrite=True,
        )["secure_url"]
        path = build_vector_store(pdf_id, pdf_bytes, extraction_mode)
        db.pdfs.insert_one(
            {
                "pdf_id": pdf_id,
                "folder_id": folder_id,
                "pdf_name": file.name,
                "pdf_url": url,
                "vector_path": str(path),
                "extraction_mode": extraction_mode,
            }
        )
        return pdf_id
    finally:
        if temp.exists():
            temp.unlink()


def document_context(pdf, question):
    store = FAISS.load_local(str(ensure_vector_store(pdf)), embeddings, allow_dangerous_deserialization=True)
    docs = store.as_retriever(search_kwargs={"k": 3}).invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)


def rag_answer(pdf, question):
    context = document_context(pdf, question)
    if not context:
        return "I could not find that in the document."
    return llm.invoke(f"Answer only from context:\n{context}\n\nQuestion: {question}").content


def notes_text(pdf_id):
    note = db.notes.find_one({"pdf_id": pdf_id})
    return note.get("content", "") if note else ""


def hybrid_answer(pdf, question):
    notes = notes_text(pdf["pdf_id"]) or "No notes."
    context = document_context(pdf, question)
    return llm.invoke(
        "Use notes first, then document context. "
        "If the answer is not in either, say 'I could not find that in the provided content.'\n\n"
        f"Notes:\n{notes}\n\nDocument context:\n{context or 'No document context.'}\n\nQuestion: {question}"
    ).content


@st.cache_data(show_spinner=False)
def fetch_pdf_bytes(pdf_url):
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    return response.content


@st.cache_data(show_spinner=False)
def extract_pdf_pages(pdf_url, extraction_mode):
    temp = Path(tempfile.gettempdir()) / f"{uuid4()}.pdf"
    try:
        temp.write_bytes(fetch_pdf_bytes(pdf_url))
        return extract_pdf_pages_from_file(temp, extraction_mode)
    finally:
        if temp.exists():
            temp.unlink()


st.set_page_config(page_title="Exam Help", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{Path('styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
st.markdown("<h1>Exam Help</h1>", unsafe_allow_html=True)

db.folders.create_index([("folder_id", 1)], unique=True)
db.pdfs.create_index([("pdf_id", 1)], unique=True)
db.notes.create_index([("pdf_id", 1)], unique=True)

st.session_state.setdefault("folder_id", None)
st.session_state.setdefault("pdf_id", None)

with st.sidebar:
    st.header("Folders")
    with st.form("create_folder_form"):
        folder_name = st.text_input("New folder")
        create_folder = st.form_submit_button("Create folder")
    if create_folder and folder_name.strip():
        db.folders.insert_one({"folder_id": str(uuid4()), "folder_name": folder_name.strip()})
        st.rerun()

    folders = list(db.folders.find().sort("folder_name", 1))
    if folders:
        folder_map = {folder["folder_name"]: folder["folder_id"] for folder in folders}
        st.session_state.folder_id = folder_map[st.radio("Folder list", list(folder_map))]

        st.subheader("PDFs")
        pdfs = list(db.pdfs.find({"folder_id": st.session_state.folder_id}).sort("pdf_name", 1))
        if pdfs:
            pdf_map = {pdf["pdf_name"]: pdf["pdf_id"] for pdf in pdfs}
            st.session_state.pdf_id = pdf_map[st.radio("PDF list", list(pdf_map))]
        else:
            st.session_state.pdf_id = None

    st.subheader("Upload PDF")
    with st.form("upload_pdf_form"):
        pdf_type = st.radio("PDF type", ["Normal PDF", "Scanned PDF"], horizontal=True)
        file = st.file_uploader("Choose PDF", type="pdf")
        upload_pdf_clicked = st.form_submit_button("Upload and Index")

    if upload_pdf_clicked:
        if not st.session_state.folder_id:
            st.error("Create or select a folder first.")
        elif not file:
            st.error("Choose a PDF first.")
        else:
            extraction_mode = "scanned" if pdf_type == "Scanned PDF" else "normal"
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_id = upload_pdf(file, st.session_state.folder_id, extraction_mode)
            st.rerun()

left, right = st.columns([1.55, 1.05], gap="medium")

with left:
    if not st.session_state.pdf_id:
        st.info("Create/select a folder and upload a PDF.")
    else:
        pdf = db.pdfs.find_one({"pdf_id": st.session_state.pdf_id})
        st.subheader(pdf["pdf_name"])

        with st.form("qa_form"):
            mode = st.radio("Mode", ["RAG", "Hybrid"], horizontal=True)
            question = st.text_input("Ask a question")
            get_answer = st.form_submit_button("Get answer")

        if get_answer:
            if question.strip():
                with st.spinner("Thinking..."):
                    st.session_state["answer"] = (
                        rag_answer(pdf, question) if mode == "RAG" else hybrid_answer(pdf, question)
                    )
            else:
                st.warning("Enter a question.")
        st.text_area("Answer", value=st.session_state.get("answer", ""), height=160)

        st.subheader("Notes")
        with st.form("save_note_form", clear_on_submit=True):
            note = st.text_area("Add note", height=100)
            save_note = st.form_submit_button("Save note")

        if save_note and note.strip():
            existing = notes_text(pdf["pdf_id"])
            content = f"{existing}\n\n{note.strip()}" if existing else note.strip()
            db.notes.update_one(
                {"pdf_id": pdf["pdf_id"]},
                {"$set": {"content": content, "pdf_id": pdf["pdf_id"]}, "$setOnInsert": {"note_id": str(uuid4())}},
                upsert=True,
            )
            st.rerun()
        st.text_area("Saved notes", value=notes_text(pdf["pdf_id"]), height=140, disabled=True)

with right:
    if st.session_state.pdf_id:
        pdf = db.pdfs.find_one({"pdf_id": st.session_state.pdf_id})
        if pdf and pdf.get("pdf_url"):
            st.subheader("Document Content")
            try:
                pages = extract_pdf_pages(pdf["pdf_url"], pdf.get("extraction_mode", "normal"))
                if pages:
                    page_no = st.selectbox("Page", options=list(range(1, len(pages) + 1)), index=0)
                    st.text_area(
                        "Extracted text",
                        value=pages[page_no - 1] or "No readable text found on this page.",
                        height=700,
                        disabled=True,
                    )
                else:
                    st.info("No readable text found in this PDF.")
            except Exception:
                st.warning("Document content could not be loaded here. Open the PDF directly using the link below.")
            st.markdown(f"[Open PDF directly]({pdf['pdf_url']})")
