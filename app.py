from pathlib import Path

import streamlit as st

from citewise.chunking import load_text_documents
from citewise.retriever import EvidenceRetriever


DOCUMENT_DIR = Path("data/documents")


@st.cache_resource
def build_retriever() -> tuple[EvidenceRetriever, int]:
    documents = load_text_documents(DOCUMENT_DIR)
    retriever = EvidenceRetriever()
    chunk_count = retriever.fit(documents)
    return retriever, chunk_count


st.set_page_config(
    page_title="CiteWise AI",
    page_icon="🔎",
    layout="wide",
)

st.title("CiteWise AI")
st.caption(
    "Evidence-first local document search with inspectable citations."
)

try:
    retriever, chunk_count = build_retriever()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.info(f"Indexed {chunk_count} passages from `{DOCUMENT_DIR}`.")

selected_sources = st.multiselect(
    "Sources to search",
    options=list(retriever.available_sources),
    default=list(retriever.available_sources),
    help="Restrict search results to selected documents.",
)

query = st.text_input(
    "Ask a question",
    placeholder="Example: Why should retrieval systems be evaluated?",
)

top_k = st.slider(
    "Number of evidence passages",
    min_value=1,
    max_value=10,
    value=5,
)

if query:
    if not selected_sources:
        st.warning("Select at least one document source.")
        st.stop()

    results = retriever.search(
        query,
        top_k=top_k,
        sources=selected_sources,
    )

    if not results:
        st.warning("No sufficiently relevant evidence was found.")
    else:
        st.subheader("Evidence")

        for result in results:
            with st.container(border=True):
                st.markdown(f"**{result.citation}**")
                st.write(result.text)
                st.caption(f"Relevance score: {result.score:.3f}")
