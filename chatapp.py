import html
import json
import os
import re

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_CHAT_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL",
    "models/gemini-embedding-001",
)

INDEX_DIR = "faiss_index"
INDEX_FILES = ("index.faiss", "index.pkl")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

            :root {
                --bg: #0a1220;
                --text: #e5eefb;
                --muted: #93a4bc;
                --border: rgba(148, 163, 184, 0.16);
            }

            html, body, [class*="css"] {
                font-family: "Manrope", sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(79, 209, 197, 0.14), transparent 24%),
                    linear-gradient(180deg, #08111f 0%, #0b1324 46%, #0e1728 100%);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(8, 15, 28, 0.98), rgba(13, 23, 39, 0.98));
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stCaption,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] div {
                color: var(--text);
            }

            [data-testid="stFileUploaderDropzone"] {
                background: rgba(15, 23, 42, 0.85);
                border: 1px dashed rgba(148, 163, 184, 0.35);
                border-radius: 20px;
            }

            .stTextInput > div > div {
                background: rgba(15, 23, 42, 0.82);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
            }

            .stTextInput input {
                color: var(--text);
                min-height: 3.5rem;
                padding-left: 1rem !important;
            }

            .stTextInput input::placeholder {
                color: #8fa5c0;
            }

            [data-testid="InputInstructions"] {
                display: none;
            }

            .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #0f766e, #0891b2);
                color: white;
                border: none;
                border-radius: 14px;
                padding: 0.8rem 1rem;
                font-weight: 700;
                letter-spacing: 0.01em;
                box-shadow: 0 16px 32px rgba(8, 145, 178, 0.22);
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, #0d9488, #0284c7);
            }

            .hero-card,
            .info-card,
            .response-card,
            .sidebar-card,
            .result-shell {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.80));
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow: 0 20px 60px rgba(2, 6, 23, 0.28);
            }

            .hero-card {
                padding: 2rem;
                margin-bottom: 1.2rem;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(79, 209, 197, 0.12);
                border: 1px solid rgba(79, 209, 197, 0.25);
                color: #9be7df;
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .hero-title {
                font-size: clamp(2rem, 4vw, 3.1rem);
                font-weight: 800;
                line-height: 1.05;
                margin: 1rem 0 0.7rem;
                color: #f8fbff;
            }

            .hero-copy,
            .section-copy,
            .sidebar-copy {
                color: var(--muted);
                line-height: 1.72;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.9rem;
                margin-top: 1.4rem;
            }

            .stat-chip {
                padding: 1rem;
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(148, 163, 184, 0.12);
            }

            .stat-label {
                display: block;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #8da2bd;
                margin-bottom: 0.35rem;
            }

            .stat-value {
                color: #f8fbff;
                font-size: 1.05rem;
                font-weight: 700;
            }

            .info-card,
            .response-card,
            .sidebar-card,
            .result-shell {
                padding: 1.2rem 1.25rem;
            }

            .section-title,
            .sidebar-title {
                color: #f8fbff;
                font-weight: 800;
                font-size: 1.08rem;
                margin-bottom: 0.4rem;
            }

            .question-helper {
                margin-top: 0.85rem;
                color: #7f93ad;
                font-size: 0.92rem;
            }

            .response-label,
            .result-kicker {
                color: #8fb7d9;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                margin-bottom: 0.55rem;
            }

            .result-question {
                color: #f8fbff;
                font-size: 1.05rem;
                font-weight: 700;
                line-height: 1.6;
                margin: 0;
            }

            .result-meta {
                margin-top: 0.8rem;
                color: #7f93ad;
                font-size: 0.9rem;
            }

            .sidebar-list {
                margin: 0;
                padding-left: 1rem;
                color: #d6e1ef;
                line-height: 1.7;
            }

            @media (max-width: 900px) {
                .stats-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def resolve_embedding_model() -> str:
    configured_model = DEFAULT_EMBEDDING_MODEL.strip()
    legacy_aliases = {
        "embedding-001": "models/gemini-embedding-001",
        "models/embedding-001": "models/gemini-embedding-001",
        "gemini-embedding-001": "models/gemini-embedding-001",
    }
    return legacy_aliases.get(configured_model, configured_model)


def extract_pdf_pages(pdf_docs):
    pages = []
    for pdf in pdf_docs:
        source_name = getattr(pdf, "name", "Uploaded PDF")
        reader = PdfReader(pdf)
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(
                    {
                        "source": source_name,
                        "page_number": page_number,
                        "text": text,
                    }
                )
    return pages


def get_text_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000,
        chunk_overlap=1000,
    )

    texts = []
    metadatas = []
    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            chunk = chunk.strip()
            if not chunk:
                continue
            texts.append(chunk)
            metadatas.append(
                {
                    "source": page["source"],
                    "page_number": page["page_number"],
                }
            )
    return texts, metadatas


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=resolve_embedding_model())


def save_knowledge_base(pages):
    texts, metadatas = get_text_chunks(pages)
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vector_store.save_local(INDEX_DIR)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as manifest_file:
        json.dump(
            {"pages": pages, "chunk_count": len(texts)},
            manifest_file,
            ensure_ascii=False,
            indent=2,
        )


def get_knowledge_signature():
    file_paths = [os.path.join(INDEX_DIR, filename) for filename in INDEX_FILES]
    if not all(os.path.exists(path) for path in file_paths):
        return None

    signature = [os.path.getmtime(path) for path in file_paths]
    signature.append(os.path.getmtime(MANIFEST_PATH) if os.path.exists(MANIFEST_PATH) else 0)
    return tuple(signature)


@st.cache_resource(show_spinner=False)
def load_vector_store(signature):
    embeddings = get_embeddings()
    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_data(show_spinner=False)
def load_manifest(signature):
    if not os.path.exists(MANIFEST_PATH):
        return {"pages": []}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


@st.cache_resource(show_spinner=False)
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model=DEFAULT_CHAT_MODEL,
        temperature=0.3,
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def finalize_answer(answer: str) -> str:
    cleaned = answer.strip()
    if not cleaned:
        return "answer is not available in the context"
    if cleaned.lower() == "answer is not available in the context":
        return cleaned
    return cleaned


@st.cache_data(show_spinner=False, ttl=600)
def get_answer_for_question(signature, question: str) -> str:
    vector_store = load_vector_store(signature)
    docs = vector_store.similarity_search(question)

    if not docs:
        return "answer is not available in the context"

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True,
    )
    return finalize_answer(response.get("output_text", ""))


def ask_question(question: str) -> str:
    signature = get_knowledge_signature()
    if signature is None:
        raise FileNotFoundError("Knowledge base not found. Build it from the sidebar first.")
    return get_answer_for_question(signature, question)


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">AI-powered knowledge workspace</div>
            <div class="hero-title">Multi-PDF Research Assistant</div>
            <p class="hero-copy">
                Upload one or more PDF files, build a searchable knowledge base, and ask
                focused questions against the indexed content. Designed for study, analysis,
                and document review workflows.
            </p>
            <div class="stats-grid">
                <div class="stat-chip">
                    <span class="stat-label">Retrieval</span>
                    <span class="stat-value">Page-aware semantic search</span>
                </div>
                <div class="stat-chip">
                    <span class="stat-label">Model Stack</span>
                    <span class="stat-value">Gemini chat and embeddings</span>
                </div>
                <div class="stat-chip">
                    <span class="stat-label">Use Case</span>
                    <span class="stat-value">Research, review, and Q&A</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        if os.path.exists("img/Robot.jpg"):
            st.image("img/Robot.jpg", use_container_width=True)

        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-title">Document Processing</div>
                <p class="sidebar-copy">
                    Upload your PDF files, create the vector index, and then ask questions
                    from the main workspace.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Build Knowledge Base"):
            if not pdf_docs:
                st.warning("Upload at least one PDF file before processing.")
            else:
                with st.spinner("Indexing documents..."):
                    try:
                        pages = extract_pdf_pages(pdf_docs)
                        if not pages:
                            st.error("No extractable text was found in the uploaded PDFs.")
                        else:
                            save_knowledge_base(pages)
                            st.success(
                                f"Knowledge base created successfully for {len(pdf_docs)} file(s) and {len(pages)} page(s)."
                            )
                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")

        st.markdown(
            """
            <div class="sidebar-card" style="margin-top: 1rem;">
                <div class="sidebar-title">Recommended Workflow</div>
                <ul class="sidebar-list">
                    <li>Upload PDFs and build the knowledge base.</li>
                    <li>Ask precise questions in the main panel.</li>
                    <li>Use page numbers for the most accurate answers.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_latest_result(item):
    st.markdown(
        f"""
        <div class="result-shell">
            <div class="result-kicker">Latest Result</div>
            <p class="result-question">{html.escape(item["question"])}</p>
            <div class="result-meta">Grounded in the indexed PDF content</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(item["answer"])


def main():
    st.set_page_config(
        page_title="Multi-PDF Research Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    render_sidebar()
    render_hero()

    if "history" not in st.session_state:
        st.session_state.history = []

    left_col, right_col = st.columns([1.5, 1], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-title">Ask a Question</div>
                <p class="section-copy">
                    Enter a question about the uploaded documents. The assistant will search
                    the indexed context and return a grounded answer from the PDF content.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("question_form", clear_on_submit=True):
            user_question = st.text_input(
                "Question",
                key="question_input",
                placeholder="Enter your question about the uploaded PDFs",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="question-helper">Example: Summarize the placement guidelines, explain the debarring policy, or tell me what is written on page 3.</div>',
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("Ask Assistant")

        if submitted:
            if not get_knowledge_signature():
                st.warning("Build the knowledge base first from the sidebar before asking questions.")
            elif not user_question.strip():
                st.warning("Enter a question before submitting.")
            else:
                with st.spinner("Analyzing documents..."):
                    try:
                        answer = ask_question(user_question)
                        st.session_state.history.append(
                            {"question": user_question, "answer": answer}
                        )
                    except Exception as exc:
                        st.error(f"Unable to answer the question: {exc}")

        if st.session_state.history:
            render_latest_result(st.session_state.history[-1])
            if len(st.session_state.history) > 1:
                with st.expander("Previous Answers", expanded=False):
                    for item in reversed(st.session_state.history[:-1]):
                        st.markdown(f"**Question:** {html.escape(item['question'])}")
                        st.markdown(item["answer"])
                        st.divider()
        else:
            st.markdown(
                """
                <div class="response-card">
                    <div class="response-label">Ready for Questions</div>
                    <div class="section-copy">
                        Upload PDFs, build the knowledge base, and ask your first question.
                        Results will appear here in a clean readable format.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-title">What This Workspace Does</div>
                <p class="section-copy">
                    The app extracts text from multiple PDFs, stores embeddings in a FAISS
                    index, and answers questions by grounding the response in retrieved page-aware context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card" style="margin-top: 1rem;">
                <div class="section-title">Best Practices</div>
                <p class="section-copy">
                    Ask specific questions for the strongest results. If you need a particular
                    page, include the page number in your prompt. Rebuild the knowledge base
                    whenever you upload different PDFs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
