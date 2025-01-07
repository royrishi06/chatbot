from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_embeddings(text):
    """Generates FAISS vectorstore from text."""
    # Step 1: Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Step 2: Use HuggingFaceEmbeddings wrapper
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Wrapper for SentenceTransformer
    vectorstore = FAISS.from_texts(chunks, embeddings_model)  # Correct usage of FAISS with embeddings

    return vectorstore
