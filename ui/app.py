import gradio as gr
from langchain_core.messages import HumanMessage
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.corag import process_document
from src.deepseek_llm import DeepSeekLLM

# Create LLM instance
model = DeepSeekLLM()

def upload_and_query(file_obj, query):
    if not query or file_obj is None:
        return "Please provide both a document and a query."
    
    try:
        # Handle file upload
        if hasattr(file_obj, 'name'):
            with open(file_obj.name, "rb") as f:
                file_bytes = f.read()
        else:
            return "Invalid file format"
        
        # Get relevant content from document
        vector_store = process_document(file_bytes)
        results = vector_store.similarity_search(query, k=2)
        context = "\n".join([doc.page_content for doc in results])

        # Create the query message
        query_with_context = f"Context: {context}\nQuestion: {query}"
        
        # Get response
        response = model.invoke([HumanMessage(content=query_with_context)])
        return response.content

    except Exception as e:
        print(f"Error: {str(e)}")
        return str(e)

# Gradio interface
iface = gr.Interface(
    fn=upload_and_query,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Document Q&A"
)

if __name__ == "__main__":
    iface.launch()
