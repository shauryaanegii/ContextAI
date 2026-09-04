# 📕 ContextAI

*An AI-powered document assistant that lets you interact with your PDFs using natural language, built with LangChain and Retrieval-Augmented Generation (RAG).*

## ✨ Features

- 📄 **PDF Chat** — Upload a PDF directly through the chat interface and ask questions about its content
- 🔎 **Retrieval-Augmented Generation (RAG)** — Retrieves relevant information from your document before generating an answer
- 🧠 **Semantic Search** — Uses Hugging Face embeddings to find the most relevant parts of your document
- ⚡ **Powered by Groq** — Uses a fast LLM to generate natural and easy-to-understand responses
- 💬 **Interactive Chat** — Have a conversation with your document instead of searching through pages manually
- 📥 **Download Chat** — Download your complete conversation as a text file
- 🔄 **New Chat** — Start a fresh conversation while keeping the uploaded document
- 📕 **New Document** — Clear the current document and upload another PDF

## 🛠️ Tech Stack

- 🐍 **Python** — Core programming language
- 🎨 **Streamlit** — Interactive web interface
- 🔗 **LangChain** — RAG pipeline and document processing
- ⚡ **Groq** — LLM inference
- 🤗 **Hugging Face** — Sentence Transformer embeddings
- 🗄️ **ChromaDB** — Vector database for storing document embeddings
- 📄 **PyPDF** — PDF text extraction

## 🚀 Running Locally

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/shauryaanegii/ContextAI.git
cd ContextAI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
