from transformers import pipeline

def generate_response(context, question, model_name="allenai/led-large-16384-arxiv", max_length=512):
    """Generates a response from a Hugging Face model given a context and question."""
    llm_input = f"Summarize or answer based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{question}"
    qa_pipeline = pipeline("text2text-generation", model=model_name, tokenizer=model_name)
    response = qa_pipeline(llm_input, max_length=max_length, truncation=True)
    return response[0]['generated_text']
