import os
import tempfile
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Define a cache directory for persisted vector stores
CACHE_DIR = "data/vector_cache"

def get_file_hash(file_bytes: bytes) -> str:
    """
    Computes a SHA256 hash of the file bytes.
    """
    hash_obj = hashlib.sha256(file_bytes)
    return hash_obj.hexdigest()

def process_document(file_bytes: bytes) -> Chroma:
    """
    Processes an uploaded document (in bytes):
      1. Computes a unique hash for caching.
      2. Checks if a cached vector store exists.
      3. If not cached, saves the file to a temporary location.
      4. Loads the document using PyPDFLoader.
      5. Splits the document into chunks.
      6. Generates embeddings for each chunk.
      7. Persists the embeddings in a Chroma vector store.
    
    Returns:
        A Chroma vector store containing the document chunks.
    """
    try:
        # Compute file hash for caching
        file_hash = get_file_hash(file_bytes)
        persist_path = os.path.join(CACHE_DIR, file_hash)
        os.makedirs(persist_path, exist_ok=True)
        
        # Check if cached vector store exists (if the persist directory is non-empty)
        if os.listdir(persist_path):
            try:
                vector_store = Chroma(
                    embedding_function=OpenAIEmbeddings(),
                    persist_directory=persist_path,
                    collection_name="document_collection"
                )
                print("Loaded vector store from cache.")
                return vector_store
            except Exception as load_err:
                print(f"Error loading cached vector store: {load_err}. Reprocessing document.")
        
        # Write the uploaded bytes to a temporary file (assuming PDF for now)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Load the document using PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        try:
            documents = loader.load()  # List of Document objects
        except Exception as load_doc_err:
            print(f"Error loading document: {load_doc_err}")
            raise
        
        # Split the document into chunks with some overlap to maintain context
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        # Generate embeddings for each chunk using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        # Create a new vector store with the persist_directory set to our cache folder
        vector_store = Chroma.from_documents(
            docs,
            embeddings,
            collection_name="document_collection",
            persist_directory=persist_path
        )

        # Persist the vector store to disk
        vector_store.persist()
        print("Processed document and persisted vector store to cache.")
        return vector_store

    except Exception as e:
        print(f"Error during document processing: {e}")
        raise

# Test the document processing function when running this module directly.
if __name__ == "__main__":
    try:
        with open("data/uploads/sample.pdf", "rb") as f:
            file_bytes = f.read()
        store = process_document(file_bytes)
        print("Document processed and vector store created successfully.")
    except Exception as e:
        print(f"Error during processing: {e}")
