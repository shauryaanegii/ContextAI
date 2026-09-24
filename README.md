# 📕 ContextAI

ContextAI is a Retrieval-Augmented Generation (RAG) based PDF question-answering application. It allows users to upload a PDF and interact with its content through natural language questions.


## *Features*

- 📄 Upload PDF documents directly through the chat interface
- 💬 Ask questions about the uploaded PDF using natural language
- 🔎 Semantic search using vector embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 📚 Page-aware document retrieval
- 💾 Maintains chat history during the session
- ⚡ Uses Hugging Face embeddings for document representation
- 🗄️ Uses ChromaDB as the vector database
- 🤖 Uses NVIDIA Nemotron 3 Super through OpenRouter
- 🚫 Prevents answers based on information outside the uploaded document

## *How it works*

1. **PDF processing** — `pypdf` extracts text from each page of the uploaded PDF.
2. **Chunking** — Text is split into overlapping chunks using `RecursiveCharacterTextSplitter` (chunk size 1500, overlap 100).
3. **Embeddings** — Chunks are embedded with the `sentence-transformers/all-MiniLM-L6-v2` model via `langchain_huggingface`.
4. **Vector store** — Embeddings are stored in an in-memory `Chroma` vector database.
5. **Retrieval** — On each question, the top 3 most relevant chunks are retrieved via similarity search.
6. **Answer generation** — The retrieved context and question are sent to an LLM (via OpenRouter) with instructions to answer using only the provided context.

## *Tech stack*

- [Streamlit](https://streamlit.io/) — UI and chat interface
- [pypdf](https://pypi.org/project/pypdf/) — PDF text extraction
- [LangChain](https://www.langchain.com/) — document handling, text splitting, and orchestration
- [langchain-huggingface](https://pypi.org/project/langchain-huggingface/) — embedding model integration
- [langchain-chroma](https://pypi.org/project/langchain-chroma/) — vector store
- [langchain-openrouter](https://pypi.org/project/langchain-openrouter/) — LLM access via [OpenRouter](https://openrouter.ai/)
- Model: `nvidia/nemotron-3-super-120b-a12b:free`

## *Setup*

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

You can get an API key from [OpenRouter](https://openrouter.ai/).

### 5. Run the app

```bash
streamlit run main.py
```



