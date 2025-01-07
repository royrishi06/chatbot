import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

st.title("PDF Question Answering Chatbot")
st.sidebar.header("Upload PDF")

# PDF Upload
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Extracting Text from PDF
    reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    st.write("**Extracted Text:**")
    with st.expander("View extracted text"):
        st.text_area("PDF Content", value=pdf_text, height=300)

    # Splitting Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(pdf_text)

    if not chunks:
        st.error("No text chunks found in the document. Ensure the document contains readable text.")
    else:
        # Generating Embeddings and Creating FAISS Index
        st.write("Generating embeddings...")
        embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        embeddings = embedding_model.embed_documents(chunks)

        if embeddings is None or len(embeddings) == 0:
            st.error("Failed to generate embeddings. Please check the document and model.")
        else:
            # Ensure embeddings and chunks are paired correctly
            try:
                if len(chunks) != len(embeddings):
                    raise ValueError("Mismatch between the number of chunks and embeddings. Ensure that each chunk has a corresponding embedding.")
                vectorstore = FAISS.from_texts(chunks, embedding_model)

                # Accepting User Query
                st.write("### Ask questions about the document")
                user_question = st.text_input("Enter your question:")

                if user_question:
                    # Retrieving Relevant Chunks
                    retriever = vectorstore.as_retriever()
                    relevant_chunks = retriever.get_relevant_documents(user_question)

                    # Preparing Input for Hugging Face LLM
                    retrieved_context = "\n".join([chunk.page_content for chunk in relevant_chunks])
                    llm_input = f"Summarize or answer based on the context below:\n\nContext:\n{retrieved_context}\n\nQuestion:\n{user_question}"

                    # Generating Response Using Hugging Face LLM
                    st.write("Generating response...")
                    qa_pipeline = pipeline("text2text-generation", model="allenai/led-large-16384-arxiv")
                    response = qa_pipeline(llm_input, max_length=512, truncation=True)
                    st.write("**Answer:**", response[0]['generated_text'])
            except ValueError as e:
                st.error(f"Error processing embeddings and texts: {e}")
else:
    st.write("Upload a PDF to get started.")
