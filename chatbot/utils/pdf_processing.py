from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
