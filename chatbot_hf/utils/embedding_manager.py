from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Generates embeddings for text chunks using HuggingFaceEmbeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model.embed_documents(chunks)

def create_vectorstore(chunks, embedding_model):
    """Creates a FAISS vectorstore from chunks and their embeddings."""
    if not chunks:
        raise ValueError("No chunks provided for vectorstore creation.")
    return FAISS.from_texts(chunks, embedding_model)
