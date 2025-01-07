from transformers import pipeline

def load_qa_model():
    """Loads a free Q&A model from Hugging Face."""
    model_name = "google/flan-t5-small"  # Free Q&A model
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return qa_pipeline

def answer_question(qa_pipeline, question, context):
    """Uses the pipeline to answer questions based on the context."""
    response = qa_pipeline(question=question, context=context)
    return response["answer"]
