import streamlit as st
from PyPDF2 import PdfReader

from chunking import chunk_text
from embedding_service import generate_embedding
from vector_store import store_embeddings, search_embeddings
from rag_service import generate_answer


st.set_page_config(
    page_title="RAG IT",
    page_icon="📄",
    layout="wide"
)


st.title("📄 RAG IT")
st.subheader("AI-Powered Resume Assistant")

st.write(
    "Upload your resume and ask questions using "
    "Retrieval-Augmented Generation (RAG)."
)


# -------------------------------
# Upload Resume
# -------------------------------

st.sidebar.header("📤 Upload Resume")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF resume",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.sidebar.button("Process Resume"):

        try:

            with st.spinner("Processing resume..."):

                # 1. Read PDF
                reader = PdfReader(uploaded_file)

                text = ""

                for page in reader.pages:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text

                # 2. Chunk text
                chunks = chunk_text(text)

                # 3. Generate embeddings
                embeddings = []

                progress_bar = st.progress(0)

                for i, chunk in enumerate(chunks):

                    vector = generate_embedding(chunk)

                    embeddings.append(vector)

                    progress_bar.progress(
                        (i + 1) / len(chunks)
                    )

                # 4. Store in ChromaDB
                store_embeddings(
                    chunks,
                    embeddings
                )

                # Save information in session
                st.session_state.resume_uploaded = True
                st.session_state.resume_filename = uploaded_file.name
                st.session_state.chunks = chunks

            st.sidebar.success(
                "✅ Resume processed successfully!"
            )

        except Exception as e:

            st.sidebar.error(
                f"❌ Error: {str(e)}"
            )


# -------------------------------
# Ask Questions
# -------------------------------

st.header("❓ Ask Questions About Your Resume")


if st.session_state.get("resume_uploaded", False):

    st.info(
        "You can ask questions such as: "
        "'What are my technical skills?'"
    )

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("🔍 Search & Answer"):

        if question:

            try:

                with st.spinner("🤖 Finding answer..."):

                    # 1. Embed question
                    question_embedding = generate_embedding(
                        question
                    )

                    # 2. Search ChromaDB
                    results = search_embeddings(
                        question_embedding,
                        top_k=3
                    )

                    # 3. Get relevant chunks
                    relevant_chunks = results["documents"][0]

                    context = "\n\n".join(
                        relevant_chunks
                    )

                    # 4. Generate answer
                    answer = generate_answer(
                        question,
                        context
                    )

                st.subheader("✨ Answer")

                st.write(answer)

                # Show source context
                with st.expander(
                    "📚 View Source Context"
                ):

                    for i, chunk in enumerate(
                        relevant_chunks,
                        1
                    ):

                        st.markdown(
                            f"**Chunk {i}:**"
                        )

                        st.write(chunk)

            except Exception as e:

                st.error(
                    f"❌ Error generating answer: {str(e)}"
                )

        else:

            st.warning(
                "Please enter a question."
            )

else:

    st.info(
        "📌 Upload a PDF resume from the sidebar "
        "and click 'Process Resume' to begin."
    )