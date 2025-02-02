# DeepCORAG: A Retrieval-Augmented Generation and Chain-of-Retrieval Application

DeepCORAG is a prototype application that demonstrates an iterative retrieval-augmented generation system (CORAG) for handling complex queries over unstructured documents. This project integrates document processing, vector storage with caching, and an iterative retrieval mechanism to build richer context for generating accurate answers.

## Features

- **Document Upload and Processing:** Automatically extract text from uploaded PDFs, split them into chunks, and convert each chunk into vector embeddings.
- **Persistent Vector Store with Caching:** Stores embeddings in a persistent vector database (using Chroma) so that repeated uploads of the same document are processed quickly.
- **Iterative Retrieval (CORAG):** Uses a chain-of-retrieval process to refine the context and generate comprehensive answers for complex, multi-hop queries.
- **Extensible Architecture:** Designed to integrate with different LLMs (e.g., DeepSeek or OpenAI) and UI frameworks (e.g., Gradio) for future enhancements.
- **Robust Error Handling:** Includes try/except blocks to ensure reliable processing.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jerryola1/DeepCORAG.git
   cd DeepCORAG
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
