"""
Streamlit UI for the RAG assistant (Task 2 - Productionization).

Talks to the FastAPI backend (service/api.py) over HTTP - it holds no
RAG/LLM logic itself, so it can be scaled, redeployed, or swapped for a
different frontend (React, Gradio) independently of the backend.

Run directly for local dev:
    uv run streamlit run ui/app.py

Configure the backend URL via the RAG_API_URL environment variable
(defaults to http://localhost:8000, or http://backend:8000 inside Docker
Compose - see docker-compose.yml).
"""
import os
import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="centered")
st.title("📚 RAG Document Assistant")
st.caption(f"Backend: {API_URL}")

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "fixed", "semantic", "structural"],
        index=0,
        help="Which pre-built vector index to query. Build it first with "
             "`python -m rag_app build --strategy <name>`.",
    )
    model = st.selectbox(
        "Model",
        ["gemini-primary", "gemini-fallback", "local-model"],
        index=0,
        help="gemini-primary/fallback require GEMINI_API_KEY. local-model "
             "requires a vLLM server running (see scripts/serve_local_model.sh).",
    )
    enable_tools = st.checkbox(
        "Enable tool calling",
        value=False,
        help="Lets the model call get_document_stats() when relevant.",
    )
    top_k = st.slider("Sources to retrieve (top-k)", min_value=1, max_value=10, value=5)

    st.divider()
    if st.button("Check backend health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.ok:
                st.success("Backend is healthy.")
            else:
                st.error(f"Backend returned {r.status_code}")
        except requests.RequestException as e:
            st.error(f"Cannot reach backend: {e}")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    if turn.get("answer") is None:
        continue
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        cols = st.columns(2)
        cols[0].metric("Confidence", f"{turn['confidence']:.2f}")
        cols[1].metric("Cached", "Yes" if turn.get("cached") else "No")
        if turn.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(turn["sources"], 1):
                    st.write(f"{i}. {s}")
        if turn.get("reasoning"):
            with st.expander("Reasoning / diagnostics"):
                st.write(turn["reasoning"])

question = st.chat_input("Ask a question about the indexed document...")

if question:
    st.session_state.history.append({"question": question, "answer": None})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "strategy": strategy,
                        "model": model,
                        "top_k": top_k,
                        "enable_tools": enable_tools,
                    },
                    timeout=180,
                )
                if resp.status_code == 429:
                    st.error("Rate limit exceeded. Please wait a moment and try again.")
                    st.session_state.history.pop()
                    st.stop()
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"Could not reach the backend at {API_URL}: {e}")
                st.session_state.history.pop()
                st.stop()

        st.write(data["answer"])
        cols = st.columns(2)
        cols[0].metric("Confidence", f"{data['confidence']:.2f}")
        cols[1].metric("Cached", "Yes" if data.get("cached") else "No")
        if data.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(data["sources"], 1):
                    st.write(f"{i}. {s}")
        if data.get("reasoning"):
            with st.expander("Reasoning / diagnostics"):
                st.write(data["reasoning"])

        # Update the last history entry with the full response
        st.session_state.history[-1] = {"question": question, **data}
