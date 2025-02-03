import gradio as gr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.corag import process_document

def upload_and_query(file_obj, query):
    #check if a file was uploaded.
    if file_obj is None:
        return "Please upload a document."
    try:
        #read file bytes from the uploaded file object.
        file_bytes = file_obj.read()
    except Exception as e:
        return f"Error reading file: {e}"

    try:
        #process the document: this extracts text, splits into chunks,
        #generates embeddings, and creates (or loads) a persisted vector store.
        vector_store = process_document(file_bytes)
    except Exception as e:
        return f"Error processing document: {e}"
    
    try:
        #retrieve relevant document chunks based on the query.
        #this performs a similarity search on the vector store.
        results = vector_store.similarity_search(query)
        #combine the retrieved chunks for display.
        retrieved_text = "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Error during retrieval: {e}"
    
    return retrieved_text

#gradio interface: allows file upload and text query
iface = gr.Interface(
    fn=upload_and_query,
    inputs=[
        gr.components.File(label="Upload Document (PDF)"),
        gr.components.Textbox(lines=2, placeholder="Enter your query here", label="Query")
    ],
    outputs=gr.components.Textbox(label="Retrieved Information"),
    title="DeepCORAG Document Q&A",
    description="Upload a PDF document and ask a question. The system will process the document, cache its embeddings, and retrieve relevant information using an iterative CORAG approach."
)

if __name__ == "__main__":
    iface.launch()
