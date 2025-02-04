import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.corag import process_document


#create an LLM instance for iterative query refinement 
model = ChatOpenAI(model="gpt-4o", temperature=0)

def upload_and_query(file_obj, query):
    #check if a file was uploaded.
    if file_obj is None:
        return "Please upload a document."
    
    try:
        #updated file reading logic to handle different file_obj types
        if isinstance(file_obj, str):
            with open(file_obj, "rb") as f:
                file_bytes = f.read()
        elif isinstance(file_obj, dict) and "name" in file_obj:
            with open(file_obj["name"], "rb") as f:
                file_bytes = f.read()
        else:
            file_bytes = file_obj.read()
    except Exception as e:
        return f"Error reading file: {e}"
    
    try:
        #process the document to generate (or load) the persistent vector store.
        vector_store = process_document(file_bytes)
    except Exception as e:
        return f"Error processing document: {e}"
    
    #initial retrieval: perform a similarity search with the user query.
    try:
        results = vector_store.similarity_search(query)
        aggregated_context = "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Error during initial retrieval: {e}"
    
    #CORAG iterative retrieval:
    max_iterations = 2  # For demonstration, we limit to 2 iterations.
    followup_prompt_template = (
        "Based on the original question and the context retrieved so far:\n"
        "Original Question: {question}\n"
        "Aggregated Context: {context}\n"
        "If this context is insufficient to answer the question fully, "
        "generate a follow-up query to retrieve additional information.\n"
        "If you have enough information, simply respond with \"Enough\".\n"
        "Follow-up Query:"
    )
    
    for i in range(max_iterations):
        prompt = followup_prompt_template.format(question=query, context=aggregated_context)
        try:
            followup_response = model.invoke([HumanMessage(content=prompt)])
            followup_query = followup_response.content.strip()
            print(f"Iteration {i+1} follow-up query: {followup_query}") 
        except Exception as e:
            return f"Error during follow-up query generation: {e}"
        
        if followup_query.lower() == "enough":
            break
        
        try:
            new_results = vector_store.similarity_search(followup_query)
            new_context = "\n\n".join([doc.page_content for doc in new_results])
            aggregated_context += "\n\n" + new_context
        except Exception as e:
            return f"Error during follow-up retrieval: {e}"
    
    #return the aggregated context as the final retrieved information.
    return aggregated_context

#gradio interface: allows file upload and text query input.
iface = gr.Interface(
    fn=upload_and_query,
    inputs=[
        gr.components.File(label="Upload Document (PDF)"),
        gr.components.Textbox(lines=2, placeholder="Enter your query here", label="Query")
    ],
    outputs=gr.components.Textbox(label="Retrieved Information"),
    title="DeepCORAG Document Q&A with Iterative Retrieval",
    description="Upload a PDF document and ask a question. The system will process the document and use an iterative CORAG approach to retrieve relevant information."
)

if __name__ == "__main__":
    iface.launch()
