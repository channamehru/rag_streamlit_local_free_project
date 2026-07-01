import streamlit as st
from rag import build_local_rag_index, answer_question


st.set_page_config(
    page_title="Local Company RAG Chatbot",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Company RAG-Based Custom Chatbot")
st.caption("Free local version: answers using only local TXT and Excel files. No OpenAI API key required.")


@st.cache_resource(show_spinner="Loading local data and building local FAISS index...")
def get_rag_index():
    """Data is loaded, chunked, vectorized, and indexed only once."""
    return build_local_rag_index()


try:
    rag_index = get_rag_index()
except Exception as e:
    st.error(f"Knowledge base loading failed: {e}")
    st.stop()


with st.sidebar:
    st.header("Project Details")
    st.write("**Version:** Free local version")
    st.write("**Data source:** `data/company_info.txt` and `data/products.xlsx`")
    st.write("**Embedding method:** Local TF-IDF vectors")
    st.write("**Vector store:** Local vector index / FAISS when available")
    st.write("**Retriever:** Top-3 chunks")
    st.write("**Fallback:** `I don't know`")
    st.info("This version does not use OpenAI API, so it works without billing credits.")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me a question about the company data."
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_question = st.chat_input("Ask a question from the local company knowledge base...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant local chunks..."):
            answer, retrieved_chunks = answer_question(rag_index, user_question)

        st.markdown(answer)

        with st.expander("Show retrieved top-3 chunks"):
            for i, item in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Chunk {i} — Source:** `{item['source']}` — Score: `{item['score']:.4f}`")
                st.write(item["text"])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
