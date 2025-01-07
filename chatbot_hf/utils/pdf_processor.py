from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(pdf_file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits the extracted text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)
