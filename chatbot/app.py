import streamlit as st
from utils.pdf_processing import extract_text_from_pdf
from utils.embeddings import create_embeddings
from utils.qa_model import load_qa_model, answer_question

st.title("PDF Question Answering with Free Models")
st.sidebar.header("Upload PDF")

# Step 1: Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Step 2: Extract Text
    st.write("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.write("**Extracted Text:**")
    with st.expander("View extracted text"):
        st.text_area("PDF Content", value=pdf_text, height=300)

    # Step 3: Generate Embeddings
    st.write("Generating embeddings...")
    vectorstore = create_embeddings(pdf_text)

    # Step 4: Load Q&A Model
    st.write("Loading the Q&A model...")
    qa_model = load_qa_model()

    # Step 5: Q&A Interface
    st.write("### Ask questions about the document")
    user_question = st.text_input("Enter your question:")
    if user_question:
        with st.spinner("Fetching answer..."):
            context = pdf_text  # Use the entire document or relevant parts
            response = answer_question(qa_model, user_question, context)
        st.write("**Answer:**", response)
else:
    st.write("Upload a PDF to get started.")
