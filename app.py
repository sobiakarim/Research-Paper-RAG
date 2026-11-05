"""
Research Paper Assistant — RAG system
Upload PDFs, get summaries, key findings, and ask questions grounded in the papers.
"""

import os
import io
import numpy as np
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHAT_MODEL = "llama-3.3-70b-versatile"  #groq
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   
CHUNK_SIZE = 1200      
CHUNK_OVERLAP = 200
TOP_K = 5              
STORE_DIR = "store"

os.makedirs(STORE_DIR, exist_ok=True)

st.set_page_config(page_title="Research Paper Assistant", layout="wide")


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

# Session state

if "papers" not in st.session_state:
    # papers: { filename: {"chunks": [...], "embeddings": np.array, "summary": str, "findings": str} }
    st.session_state.papers = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar — API key & upload

st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.environ.get("GROQ_API_KEY", ""))
st.sidebar.caption("Free key: https://console.groq.com/keys")

if not api_key:
    st.sidebar.warning("Enter your Groq API key to use the app.")
    st.stop()

client = Groq(api_key=api_key)
embedder = load_embedder()

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("Upload research papers (PDF)", type=["pdf"], accept_multiple_files=True)

# Helper functions

def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text)


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def cosine_sim(query_vec, matrix):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return m @ q


def retrieve_chunks(query, paper_names, top_k=TOP_K):
    """Retrieve top_k chunks across the selected papers."""
    q_vec = embed_texts([query])[0]
    candidates = []
    for name in paper_names:
        paper = st.session_state.papers[name]
        sims = cosine_sim(q_vec, paper["embeddings"])
        for idx, score in enumerate(sims):
            candidates.append((score, name, paper["chunks"][idx]))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:top_k]


def call_llm(system_prompt, user_prompt, max_tokens=900):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def summarize_paper(text):
    truncated = text[:15000]  # keep prompt manageable
    system = "You are an expert research assistant who writes clear, accurate summaries of academic papers."
    user = f"""Summarize the following research paper. Cover: (1) the problem/motivation, (2) the method/approach, (3) the main results, (4) limitations if mentioned. Keep it to 4-6 concise paragraphs or bullet points.

PAPER TEXT:
{truncated}"""
    return call_llm(system, user)


def extract_findings(text):
    truncated = text[:15000]
    system = "You are an expert research assistant who extracts key findings from academic papers."
    user = f"""Extract the key findings from this paper as a clear, numbered list. Include concrete results (numbers, metrics, comparisons) where available. Be specific and avoid vague statements.

PAPER TEXT:
{truncated}"""
    return call_llm(system, user)


def process_paper(filename, pdf_bytes):
    with st.spinner(f"Processing {filename}..."):
        text = extract_text(pdf_bytes)
        if not text.strip():
            st.error(f"Could not extract text from {filename}. It may be a scanned/image PDF.")
            return
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        summary = summarize_paper(text)
        findings = extract_findings(text)
        st.session_state.papers[filename] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "summary": summary,
            "findings": findings,
            "full_text": text,
        }


# Handle uploads

if uploaded_files:
    for f in uploaded_files:
        if f.name not in st.session_state.papers:
            process_paper(f.name, f.read())

# Main UI

st.title("📚 Research Paper Assistant")
st.caption("Upload papers, get summaries and key findings, then ask questions across your library.")

if not st.session_state.papers:
    st.info("Upload one or more PDF papers from the sidebar to get started.")
    st.stop()

paper_names = list(st.session_state.papers.keys())

tab_summary, tab_findings, tab_qa = st.tabs(["📝 Summaries", "🔑 Key Findings", "💬 Ask Questions"])

with tab_summary:
    for name in paper_names:
        with st.expander(f"📄 {name}", expanded=True):
            st.markdown(st.session_state.papers[name]["summary"])

with tab_findings:
    for name in paper_names:
        with st.expander(f"📄 {name}", expanded=True):
            st.markdown(st.session_state.papers[name]["findings"])

with tab_qa:
    selected = st.multiselect("Ask across which papers?", paper_names, default=paper_names)
    question = st.text_input("Your question")
    ask = st.button("Ask")

    if ask and question and selected:
        with st.spinner("Searching papers and generating answer..."):
            top_chunks = retrieve_chunks(question, selected)
            context = "\n\n---\n\n".join(
                [f"[Source: {name}]\n{chunk}" for _, name, chunk in top_chunks]
            )
            system = (
                "You are a research assistant answering questions strictly using the provided paper excerpts. "
                "Cite which paper(s) (by filename) support each claim. If the excerpts don't contain the answer, say so clearly."
            )
            user = f"""Context excerpts from the papers:

{context}

Question: {question}

Answer using only the context above. Mention source filenames where relevant."""
            answer = call_llm(system, user)
            st.session_state.chat_history.append((question, answer, [n for _, n, _ in top_chunks]))

    for q, a, sources in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.markdown(a)
        st.caption(f"Sources consulted: {', '.join(sorted(set(sources)))}")
        st.markdown("---")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear all papers"):
    st.session_state.papers = {}
    st.session_state.chat_history = []
    st.rerun()
