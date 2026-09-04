import streamlit as st

from api_client import ask_question


st.set_page_config(
    page_title="Sherlock Holmes RAG Assistant",
    page_icon="🔎",
    layout="centered"
)


st.title("🔎 Sherlock Holmes RAG Assistant")
st.write("Ask questions about The Complete Sherlock Holmes.")

with st.sidebar:
    st.header("About the Assistant")

    st.write(
        "This assistant uses Retrieval-Augmented Generation "
        "to answer questions from The Complete Sherlock Holmes."
    )

    st.markdown("### RAG Pipeline")

    st.write("📄 Document")
    st.write("✂️ Chunking")
    st.write("🧠 Embeddings")
    st.write("🔎 Hybrid Retrieval")
    st.write("🤖 Qwen 3")
    st.write("💬 Answer + Sources")

question = st.chat_input("Ask a question about Sherlock Holmes...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the book..."):
            try:
                result = ask_question(question)

                st.markdown("### Answer")
                st.write(result["answer"])

                if result["sources"]:
                    st.markdown("**Sources:**")
                    for source in result["sources"]:
                        st.write(f"- {source}")

            except Exception as e:
                st.error(f"Error: {e}")