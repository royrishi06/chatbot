<div align="center">

# Chatbot Using Hugging Face

An interactive application using Streamlit that allows users to upload a PDF, converts its content into embeddings, and enables question-answering via an LLM.
</div>

### 1. File Directory Structure:
```bash

chatbot_hf/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
└── utils/                 # Utility functions
    ├── pdf_processor.py   # Handles PDF text extraction and splitting
    ├── embedding_manager.py  # Manages embeddings and vectorstore
    └── llm_handler.py     # Handles Hugging Face LLM interaction
```
### 2. Install Dependencies
#### 1. Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```
#### 2. Install dependencies from requirements.txt:
```
pip install -r requirements.txt
```
### 3. Usage Instructions
1. Run the Application
From the project/ directory, run the following command to start the Streamlit app:

```
Copy code
streamlit run app.py
```
### Usage
#### 1. Run the Application
From the project/ directory, run the following command to start the Streamlit app:

```
streamlit run app.py
```
#### 2. Upload a PDF
Open the app in your browser (the link will be shown in the terminal, typically at http://localhost:8501).
Use the Upload PDF section in the sidebar to upload your research paper or any other document.

#### 3. Explore the PDF
After uploading:
The text extracted from the PDF will be displayed in an expandable section called Extracted Text.
The text will be split into manageable chunks (e.g., 1,000 characters with overlap).

#### 4. Ask Questions
Type a question related to the content of the uploaded document in the Ask questions about the document section.
The system will:
Retrieve the most relevant chunks from the FAISS vector store.
Pass the context and your question to the Hugging Face model (allenai/led-large-16384-arxiv).
Display the generated answer.

### Key Features
Efficient Handling of Large PDFs:
The PDF is split into manageable chunks for embedding and querying.
Interactive QA:
Ask questions about the document, and the system retrieves relevant context and answers intelligently.
Hugging Face LLM Integration:
Uses allenai/led-large-16384-arxiv to handle long documents like research papers.

### Bonus:
The Chatbot has been deployed on the platform Streamlit Cloud. The screenshots
