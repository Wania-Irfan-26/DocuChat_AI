"""
╔══════════════════════════════════════════════════════════╗
║              DocuChat AI — Streamlit RAG App             ║
║  Upload PDFs, TXTs, or Word docs and chat with them!     ║
║  v2 — White theme · Poppins · Clean bubbles              ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import re
import html
import tempfile
import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables from .env ─────────────────────────────────────
load_dotenv()

# ── LangChain imports (RAG backend — unchanged) ───────────────────────────────
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DocuChat AI",
    page_icon="💬",
    layout="wide",
    # FIX: "expanded" ensures sidebar opens on every page load,
    # including after a rerun triggered by send/reset actions.
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
#  UI FIX 1 — Theme: white background, indigo accents, gold secondary
#  UI FIX 2 — Font: Poppins 36 px bold for title, clean body font
#  UI FIX 3 — Sidebar: soft gray #F5F5F5, minimal content only
#  UI FIX 4 — Chat bubbles: user=right indigo, AI=left gold
#  UI FIX 5 — Responsive chat window with proper flex scroll
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ─────────────────────────────────────────────────
       UI FIX 2: Import Poppins from Google Fonts
       Replaces the previous Syne/DM Sans dark-theme fonts
    ───────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* ─────────────────────────────────────────────────
       UI FIX 1: Root palette — white theme
    ───────────────────────────────────────────────── */
    :root {
        --indigo:      #4B0082;
        --indigo-l:    #6A1BA3;
        --indigo-pale: #EDE7F6;   /* light indigo tint for user bubbles */
        --gold:        #FFD700;
        --gold-pale:   #FFFBEA;   /* light gold tint for AI bubbles */
        --gold-border: #F0C300;
        --bg:          #FFFFFF;   /* UI FIX 1: white background */
        --sidebar-bg:  #F5F5F5;   /* UI FIX 3: soft gray sidebar */
        --surface:     #FAFAFA;
        --border:      #E0E0E0;
        --text:        #1A1A2E;
        --muted:       #757575;
        --success-bg:  #E8F5E9;
        --success-txt: #2E7D32;
        --error-bg:    #FFEBEE;
        --error-txt:   #C62828;
        --info-bg:     #E3F2FD;
        --info-txt:    #1565C0;
    }

    /* ─────────────────────────────────────────────────
       UI FIX 1 + 2: Base — white bg, Poppins everywhere
    ───────────────────────────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* FIX: Hide the sidebar collapse/expand arrow button entirely.
       Prevents users from accidentally closing the sidebar with no way back.
       Targets every selector Streamlit uses across versions. */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"] { display: none !important; }

    /* FIX: Lock sidebar to a fixed width so it can never collapse to zero */
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 320px !important;
        width: 300px !important;
    }

    /* ─────────────────────────────────────────────────
       UI FIX 3: Sidebar — soft gray, decluttered
    ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 280px !important;
        max-width: 320px !important;
    }
    /* Make all sidebar text dark so it reads on light bg */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: var(--text) !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Sidebar section headings */
    .sb-heading {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: var(--indigo) !important;
        margin: 20px 0 8px;
    }

    /* Sidebar instruction list */
    .sb-instructions {
        font-size: 0.83rem;
        color: #444 !important;
        line-height: 1.85;
        padding-left: 18px;
        margin: 0;
    }

    /* File chip in sidebar */
    .file-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #fff;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 10px;
        margin-bottom: 6px;
        font-size: 0.8rem;
        color: var(--text);
        word-break: break-all;
    }
    .file-chip .ext-badge {
        background: var(--indigo);
        color: #fff;
        font-size: 0.62rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.5px;
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* ─────────────────────────────────────────────────
       UI FIX 2: App header — Poppins 36 px bold title
    ───────────────────────────────────────────────── */
    .app-header {
        background: linear-gradient(120deg, var(--indigo) 0%, #7B00C8 100%);
        border-radius: 14px;
        padding: 26px 32px 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(75,0,130,0.18);
    }
    .app-header h1 {
        /* UI FIX 2: Poppins 36px bold — professional clean title */
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700;
        font-size: 36px;
        color: #ffffff;
        margin: 0 0 6px;
        letter-spacing: -0.3px;
        line-height: 1.15;
    }
    .app-header h1 .gold-word { color: var(--gold); }
    .app-header .subtitle {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.78);
        margin: 0;
        font-weight: 400;
    }

    /* ─────────────────────────────────────────────────
       Status banners (success / error / info)
    ───────────────────────────────────────────────── */
    .status-banner {
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 0.87rem;
        font-weight: 500;
        margin-bottom: 14px;
        border-left: 4px solid transparent;
    }
    .status-banner.success {
        background: var(--success-bg);
        color: var(--success-txt);
        border-left-color: var(--success-txt);
    }
    .status-banner.error {
        background: var(--error-bg);
        color: var(--error-txt);
        border-left-color: var(--error-txt);
    }
    .status-banner.info {
        background: var(--info-bg);
        color: var(--info-txt);
        border-left-color: var(--info-txt);
    }

    /* ─────────────────────────────────────────────────
       UI FIX 4 + 5: Chat window — responsive, scrollable
    ───────────────────────────────────────────────── */
    .chat-window {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px 16px;
        /* UI FIX 5: viewport-relative max height so it scales on any screen */
        min-height: 300px;
        max-height: 60vh;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-bottom: 16px;
        scrollbar-width: thin;
        scrollbar-color: #D1C4E9 transparent;
    }
    .chat-window::-webkit-scrollbar { width: 4px; }
    .chat-window::-webkit-scrollbar-thumb {
        background: #D1C4E9;
        border-radius: 99px;
    }

    /* Empty-state placeholder */
    .empty-state {
        margin: auto;
        text-align: center;
        color: var(--muted);
        padding: 40px 20px;
    }
    .empty-state .icon { font-size: 2.6rem; }
    .empty-state p { font-size: 0.88rem; margin-top: 8px; }

    /* ─────────────────────────────────────────────────
       UI FIX 4: User bubble — RIGHT, light indigo tint
    ───────────────────────────────────────────────── */
    .bubble-row-user {
        display: flex;
        justify-content: flex-end;
    }
    .bubble-user {
        max-width: 70%;
        background: var(--indigo-pale);
        border: 1px solid #C5A8E8;
        color: var(--indigo);
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;  /* tail: bottom-right */
        font-size: 0.9rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(75,0,130,0.10);
    }
    .bubble-user .bubble-label {
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--indigo-l);
        margin-bottom: 4px;
    }

    /* ─────────────────────────────────────────────────
       UI FIX 4: AI bubble — LEFT, gold tint
    ───────────────────────────────────────────────── */
    .bubble-row-ai {
        display: flex;
        justify-content: flex-start;
    }
    .bubble-ai {
        max-width: 75%;
        background: var(--gold-pale);
        border: 1px solid var(--gold-border);
        color: #4A3800;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;  /* tail: bottom-left */
        font-size: 0.9rem;
        line-height: 1.65;
        box-shadow: 0 2px 8px rgba(240,195,0,0.18);
    }
    .bubble-ai .bubble-label {
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #B8860B;
        margin-bottom: 4px;
    }

    /* ── Regular buttons ── */
    [data-testid="stButton"] > button {
        background-color: var(--indigo) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        transition: background 0.2s, transform 0.15s !important;
        box-shadow: 0 2px 10px rgba(75,0,130,0.22) !important;
    }
    [data-testid="stButton"] > button *,
    [data-testid="stButton"] > button p,
    [data-testid="stButton"] > button span,
    [data-testid="stButton"] > button div {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif !important;
    }
    [data-testid="stButton"] > button:hover,
    [data-testid="stButton"] > button:hover * {
        background-color: var(--indigo-l) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* ── Form submit button (Send ➤) — Streamlit uses a different element ── */
    [data-testid="stFormSubmitButton"] > button {
        background-color: var(--indigo) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 9px 20px !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        transition: background 0.2s, transform 0.15s !important;
        box-shadow: 0 2px 10px rgba(75,0,130,0.22) !important;
        width: 100% !important;
    }
    [data-testid="stFormSubmitButton"] > button *,
    [data-testid="stFormSubmitButton"] > button p,
    [data-testid="stFormSubmitButton"] > button span,
    [data-testid="stFormSubmitButton"] > button div {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: var(--indigo-l) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover * {
        color: #ffffff !important;
    }

    /* Text input — clean white with indigo focus ring */
    [data-testid="stTextInput"] input {
        background: #ffffff !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.92rem !important;
        padding: 11px 15px !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: var(--indigo) !important;
        box-shadow: 0 0 0 3px rgba(75,0,130,0.10) !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: #BDBDBD !important; }

    /* ─────────────────────────────────────────────────
       File uploader — dark indigo background, white text
       Fix: every text node inside the uploader widget
       (the "Drag and drop" label, "Browse files" button,
       file size limit note, and any span/p/small) is
       forced to white so it reads on the dark background.
    ───────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--indigo) !important;
        border: 1.5px dashed rgba(255,255,255,0.35) !important;
        border-radius: 10px !important;
    }
    /* All text inside the uploader box */
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }
    /* "Browse files" inner button inside the uploader */
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] button * {
        background-color: rgba(255,255,255,0.15) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] button:hover * {
        background-color: rgba(255,255,255,0.28) !important;
        color: #ffffff !important;
    }
    /* Uploaded file name chip that appears after upload */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] * {
        color: #ffffff !important;
        background-color: rgba(255,255,255,0.12) !important;
    }
    /* Delete (x) button on the file chip */
    [data-testid="stFileUploaderDeleteBtn"] button,
    [data-testid="stFileUploaderDeleteBtn"] button * {
        color: #ffffff !important;
    }

    /* Divider */
    hr { border-color: var(--border) !important; margin: 14px 0 !important; }

    /* Section label above chat */
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: var(--indigo);
        margin-bottom: 10px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        margin-top: 36px;
        padding-top: 16px;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.77rem;
    }
    .app-footer b { color: var(--indigo); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "chat_history"        not in st.session_state: st.session_state.chat_history = []
if "rag_chain"           not in st.session_state: st.session_state.rag_chain = None
if "uploaded_file_names" not in st.session_state: st.session_state.uploaded_file_names = []
if "status_message"      not in st.session_state: st.session_state.status_message = None
# Integer key for the text_input widget — incrementing forces Streamlit to
# re-create the widget empty, clearing the box after each send.
if "input_key"           not in st.session_state: st.session_state.input_key = 0  # kept for reset compatibility


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND HELPERS  (RAG logic — unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

def load_single_document(file_path: str):
    """Load a document from disk. Supports .txt, .pdf, .docx."""
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "txt":
        return TextLoader(file_path, encoding="utf-8").load()
    elif ext == "pdf":
        return PyPDFLoader(file_path).load()
    elif ext == "docx":
        return Docx2txtLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def sanitize_answer(text: str) -> str:
    """
    UI FIX 4: Strip LangChain / Groq artefact tokens from AI replies.
    Removes: <<...>>, <|...|>, [INST]/[/INST], stray angle-bracket clusters,
    and triple+ newlines.
    """
    text = re.sub(r"<<[^>]*>>", "", text)           # <<TOKEN>> patterns
    text = re.sub(r"<\|[^|]*\|>", "", text)         # <|special|> tokens
    text = re.sub(r"\[/?INST\]", "", text, flags=re.IGNORECASE)  # [INST] tags
    text = re.sub(r"[<>]{3,}", "", text)             # >>>/<< clusters
    text = re.sub(r"\n{3,}", "\n\n", text)           # excessive blank lines
    return text.strip()


def build_rag_chain(uploaded_files):
    """
    Full RAG pipeline:
      1. Save each upload to a temp file
      2. Load & split into chunks
      3. Embed with HuggingFace sentence-transformers
      4. Store in in-memory Chroma vector DB
      5. Build and return a LangChain retrieval chain
    Returns: (chain | None, list[str] names, str | None warning)
    """
    all_docs, loaded_names, errors = [], [], []

    for uf in uploaded_files:
        suffix = "." + uf.name.rsplit(".", 1)[-1].lower()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            docs = load_single_document(tmp_path)
            all_docs.extend(docs)
            loaded_names.append(uf.name)
            os.unlink(tmp_path)         # clean up temp file immediately
        except Exception as e:
            errors.append(f"{uf.name}: {e}")

    if not all_docs:
        return None, [], "No valid documents loaded. " + "; ".join(errors)

    # ── Chunking ────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunked_docs = splitter.split_documents(all_docs)

    # ── Embedding ───────────────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Vector store (in-memory Chroma) ─────────────────────────────────────
    vector_store = Chroma.from_documents(documents=chunked_docs, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # ── LLM ─────────────────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return None, [], "GROQ_API_KEY not found. Add it to your .env file."

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key)

    # ── Prompt ──────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_template(
        """You are DocuChat AI, a helpful and precise assistant.
Answer the user's question ONLY based on the context below.
If the answer is not in the context, say:
"I couldn't find that in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""
    )

    # ── Chain ────────────────────────────────────────────────────────────────
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    warning = ("⚠️ Skipped: " + "; ".join(errors)) if errors else None
    return chain, loaded_names, warning


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
#  UI FIX 3: Reduced to 3 sections only — Instructions · Upload · Reset
#            Removed "About" block and any extra clutter
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    st.markdown(
        """
        <div style="padding:14px 0 4px; text-align:center;">
            <span style="font-family:'Poppins',sans-serif; font-size:1.4rem;
                         font-weight:800; color:#4B0082;">💬 DocuChat</span>
            <span style="font-family:'Poppins',sans-serif; font-size:1.4rem;
                         font-weight:800; color:#FFD700;"> AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── How to Use (UI FIX 3: short, no fluff) ────────────────────────────────
    st.markdown('<div class="sb-heading">How to Use</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ol class="sb-instructions">
          <li>Upload PDF, TXT, or DOCX files</li>
          <li>Click <b>Process Documents</b></li>
          <li>Type your question and press <b>Send</b></li>
          <li>Use <b>Reset Chat</b> to start over</li>
        </ol>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── File Upload ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-heading">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_clicked = st.button("⚡ Process Documents", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.session_state.status_message = ("error", "⚠️ Please upload at least one file first.")
        else:
            with st.spinner("Embedding documents… this may take a moment."):
                chain, names, err = build_rag_chain(uploaded_files)
            if chain:
                st.session_state.rag_chain = chain
                st.session_state.uploaded_file_names = names
                st.session_state.chat_history = []      # fresh chat for new doc set
                st.session_state.input_key += 1         # clear any leftover input
                msg = f"✅ {len(names)} file(s) indexed successfully!"
                if err: msg += f"  {err}"
                st.session_state.status_message = ("success", msg)
            else:
                st.session_state.status_message = ("error", f"❌ {err}")

    # ── Indexed file list ─────────────────────────────────────────────────────
    if st.session_state.uploaded_file_names:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sb-heading">Indexed Files</div>', unsafe_allow_html=True)
        for name in st.session_state.uploaded_file_names:
            ext = name.rsplit(".", 1)[-1].upper()
            safe_name = html.escape(name)   # prevent special chars breaking HTML
            st.markdown(
                f'<div class="file-chip">'
                f'<span class="ext-badge">{ext}</span>{safe_name}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Reset Chat button ──────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🔄 Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.input_key += 1     # UI FIX 4: clear input box on reset too
        st.session_state.status_message = ("info", "💬 Chat cleared. Documents are still indexed.")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Header (UI FIX 2: 36px Poppins bold) ─────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>💬 DocuChat <span class="gold-word">AI</span></h1>
        <p class="subtitle">
            Upload your documents and have an intelligent conversation with them — instantly.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Status banner ─────────────────────────────────────────────────────────────
if st.session_state.status_message:
    kind, text = st.session_state.status_message
    # html.escape so status text with special chars renders safely
    st.markdown(
        f'<div class="status-banner {kind}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )

# ── Chat window ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

parts = ['<div class="chat-window" id="docuchat-window">']

if not st.session_state.chat_history:
    # Empty placeholder shown before any messages
    parts.append(
        '<div class="empty-state">'
        '<div class="icon">📄</div>'
        '<p>Upload and process your documents, then ask me anything about them.</p>'
        '</div>'
    )
else:
    for msg in st.session_state.chat_history:
        # UI FIX 4: html.escape prevents doc content from injecting HTML into bubbles
        safe_text = html.escape(msg["content"]).replace("\n", "<br>")

        if msg["role"] == "user":
            # UI FIX 4: user message aligned RIGHT, light indigo bubble
            parts.append(
                f'<div class="bubble-row-user">'
                f'<div class="bubble-user">'
                f'<div class="bubble-label">You</div>'
                f'{safe_text}'
                f'</div></div>'
            )
        else:
            # UI FIX 4: AI message aligned LEFT, gold tint bubble
            parts.append(
                f'<div class="bubble-row-ai">'
                f'<div class="bubble-ai">'
                f'<div class="bubble-label">✦ DocuChat AI</div>'
                f'{safe_text}'
                f'</div></div>'
            )

parts.append("</div>")   # close .chat-window
st.markdown("".join(parts), unsafe_allow_html=True)

# ── AUTO-SCROLL ───────────────────────────────────────────────────────────────
# Streamlit strips <script> from st.markdown, so we use components.v1.html
# which renders a real iframe. We reach the parent document via window.parent
# and scroll #docuchat-window. A 300 ms delay ensures Streamlit has finished
# painting the new messages into the DOM before we scroll.
import streamlit.components.v1 as components
components.html(
    """
    <script>
      setTimeout(function() {
        try {
          var chatDiv = window.parent.document.getElementById('docuchat-window');
          if (chatDiv) {
            chatDiv.scrollTop = chatDiv.scrollHeight;
          }
        } catch(e) {}
      }, 300);
    </script>
    """,
    height=0,
    scrolling=False,
)

# ── Input + Send (st.form) ────────────────────────────────────────────────────
# WHY st.form:
#   Using a plain text_input + button causes the double-send bug because
#   Streamlit reruns on BOTH the input's on_change AND the button click —
#   two separate reruns, two messages sent.
#   st.form batches everything inside it into a SINGLE rerun that fires
#   only when the form is submitted (Enter key OR the submit button click).
#   This is the only Streamlit-native way to get Enter + button with one send.
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([8, 1.5])
    with col_input:
        user_query = st.text_input(
            label="",
            placeholder="Ask a question and press Enter or click Send…",
            label_visibility="collapsed",
        )
    with col_send:
        # form_submit_button submits the form — works on Enter key AND click
        send_clicked = st.form_submit_button("Send ➤", use_container_width=True)

# ── Handle Send ───────────────────────────────────────────────────────────────
if send_clicked:

    if not user_query or not user_query.strip():
        st.session_state.status_message = ("error", "⚠️ Please type a question before sending.")
        st.rerun()

    elif not st.session_state.rag_chain:
        st.session_state.status_message = ("error", "⚠️ Please upload and process documents first.")
        st.rerun()

    else:
        clean_query = user_query.strip()

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": clean_query})

        # Query the RAG chain
        try:
            with st.spinner("Thinking…"):
                raw_answer = st.session_state.rag_chain.invoke(clean_query)
            answer = sanitize_answer(raw_answer)
        except Exception as e:
            answer = f"⚠️ Something went wrong: {e}"

        # Append AI reply
        st.session_state.chat_history.append({"role": "ai", "content": answer})
        st.session_state.status_message = None
        st.rerun()



# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-footer">
        Built with ❤️ using <b>LangChain</b> · <b>Groq LLaMA 3.1</b> · <b>Streamlit</b>
        &nbsp;|&nbsp; DocuChat AI — Portfolio Project
    </div>
    """,
    unsafe_allow_html=True,
)