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

3. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate

pip install -r requirements.txt
   ```

4. **Activate virtual environment:**
   ```bash
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

5. **Set up your API keys:** Create a .env file in the root directory and add your keys:
   ```bash
   OPENAI_API_KEY=your_openai_key
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

6. **Run the application:**
   ```bash
   python src/corag.py
   ```

## Usage

1. **Upload a PDF document:**
   - Click the "Upload" button in the Gradio interface.
   - Select a PDF file to upload.
   - The application will process the document and display the results.

2. **Enter a query:**
   - Type your query in the input field.
   - Click the "Submit" button to get the answer.

3. **View the results:**
   - The application will display the answer and the sources used to generate the answer.

## Contributing

1. **Fork the repository:**
   - Click the "Fork" button on the top right of the repository page.

2. **Create a new branch:**
   - Click the "Branch" button on the top right of the repository page.
   - Type a name for your branch (e.g., "feature-new-feature").
   - Click the "Create branch" button.

3. **Make your changes and commit them:**
   - Click the "Commit" button on the top right of the repository page.

4. **Push your changes:**
   - Click the "Push" button on the top right of the repository page.

5. **Create a pull request:**
   - Click the "Pull request" button on the top right of the repository page.
   - Type a title for your pull request.
   - Click the "Create pull request" button.

6. **Merge your pull request:**
   - Click the "Merge" button on the top right of the repository page.

7. **Update your local repository:**
   - Click the "Pull" button on the top right of the repository page.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

info@abayomiolagunju.net 