# 📚 Research Paper Assistant (RAG)

A Streamlit app that lets you upload PDF research papers and:
- Generate a structured **summary** (problem, method, results, limitations)
- Extract **key findings** as a numbered list
- **Ask questions** across one or more papers, with answers grounded in retrieved excerpts (RAG) and source citations

## How it works
1. Each uploaded PDF is text-extracted (`pypdf`) and split into overlapping chunks.
2. Each chunk is embedded **locally and for free** using `sentence-transformers` (`all-MiniLM-L6-v2`) — no API calls, runs on your machine.
3. When you ask a question, your query is embedded the same way and compared (cosine similarity) against all chunks to retrieve the most relevant ones.
4. Those chunks are passed as context to **Groq's `llama-3.3-70b-versatile`** (free tier, very fast), which answers using only that context and cites the source paper(s).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

Get a free Groq API key at https://console.groq.com/keys (no credit card required).

## Usage
1. Enter your Groq API key in the sidebar.
2. Upload one or more PDF papers.
3. Wait for processing — summary and findings generate automatically. First run will download the small local embedding model (~80MB), which is cached after that.
4. Check the **Summaries** and **Key Findings** tabs.
5. Go to **Ask Questions**, pick which papers to search, and ask anything.

## Notes
- Works best with text-based PDFs. Scanned/image-only PDFs won't extract text (no OCR included).
- Everything is stored in memory for the session only — re-upload papers if you restart the app. Let me know if you'd like persistent storage across sessions (e.g. saving embeddings to disk) added.
- Groq's free tier has generous rate limits but isn't truly unlimited — if you hit a limit, wait a bit or check https://console.groq.com for current limits.
- Embeddings are fully local and free with no rate limits at all.
