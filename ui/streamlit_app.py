"""
ClauseIQ — Streamlit UI
Upload insurance PDFs, ask questions, see answers with source citations.

Run with:
    streamlit run ui/streamlit_app.py
"""

import os
import requests
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ClauseIQ — Insurance Policy Q&A",
    page_icon="📋",
    layout="wide",
)

# ─── Styles ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        .source-card {
            background: #f0f4ff;
            border-left: 4px solid #4C6EF5;
            padding: 0.6rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        .answer-box {
            background: #f8fffe;
            border: 1px solid #20c997;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            font-size: 1.05rem;
        }
        .metric-card {
            text-align: center;
            padding: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/document.png", width=60)
    st.title("ClauseIQ")
    st.caption("Insurance Policy RAG System")
    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("✅ Backend connected")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", health.get("num_documents", 0))
        with col2:
            st.metric("Chunks", health.get("num_chunks", 0))
    except Exception:
        st.error("❌ Backend not reachable\nStart: `uvicorn app.api.main:app --reload`")

    st.divider()

    # Document filters
    st.subheader("🔍 Filter Options")
    filter_doc_type = st.selectbox(
        "Document Type",
        ["All", "policy", "endorsement", "claim", "exclusion"],
    )
    filter_document = st.text_input("Specific Document Name (optional)")

    if filter_doc_type == "All":
        filter_doc_type = None
    if not filter_document.strip():
        filter_document = None

    st.divider()

    # Upload section
    st.subheader("📤 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file and st.button("Ingest PDF", type="primary"):
        with st.spinner("Ingesting..."):
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                data={"doc_type": "policy"},
            )
        if resp.status_code == 200:
            st.success(f"✅ {uploaded_file.name} ingested!")
            st.json(resp.json()["summary"])
            st.rerun()
        else:
            st.error(f"Upload failed: {resp.text}")

# ─── Main Area ────────────────────────────────────────────────────────────────

st.title("📋 ClauseIQ — Insurance Policy Q&A")
st.caption("Ask questions about your insurance policies. Answers are grounded in your documents with citations.")

st.divider()

# Show indexed documents
try:
    docs_resp = requests.get(f"{API_URL}/documents", timeout=3).json()
    if docs_resp.get("total", 0) > 0:
        with st.expander(f"📚 Indexed Documents ({docs_resp['total']})", expanded=False):
            for doc in docs_resp["documents"]:
                st.markdown(
                    f"**{doc['document']}** — {doc['doc_type']} "
                    f"| {doc['num_pages']} pages | {doc['num_chunks']} chunks"
                )
    else:
        st.info("No documents indexed yet. Upload PDFs using the sidebar.")
except Exception:
    pass

st.divider()

# Q&A Section
st.subheader("💬 Ask a Question")

question = st.text_area(
    "Your question:",
    placeholder="e.g. What is the deductible for flood damage? What exclusions apply to home policies?",
    height=80,
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("Ask", type="primary", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("Searching documents and generating answer..."):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "filter_doc_type": filter_doc_type,
                    "filter_document": filter_document,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                result = resp.json()

                # Answer
                st.subheader("📝 Answer")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"Model: `{result['model']}` | "
                    f"Chunks used: {result['num_chunks_used']}"
                )

                # Sources
                if result["sources"]:
                    st.subheader("📎 Sources")
                    for source in result["sources"]:
                        score_val = source.get('score', 0)
                        st.markdown(
                            f'<div class="source-card">'
                            f"📄 <b>{source['document']}</b> — "
                            f"Page {source['page']} "
                            f"({source['doc_type']}) "
                            f"| Score: {score_val:.3f}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend. Make sure the API server is running.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

elif ask_btn:
    st.warning("Please enter a question.")
