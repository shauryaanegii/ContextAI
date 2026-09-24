import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="ContextAI",
    page_icon="📕"
)
st.title("📕 ContextAI")
st.caption("Ask questions about your PDF")

# -----------------------------
# Check API key
# -----------------------------
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing from .env")
    st.stop()

# -----------------------------
# Initialize chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Embedding model
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={
            "batch_size": 32,
            "normalize_embeddings": True
        }
    )

# -----------------------------
# LLM
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenRouter(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0.1,
        max_tokens=500,
        api_key=OPENROUTER_API_KEY
    )

# -----------------------------
# Process PDF
# -----------------------------
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "page": page_number
                    }
                )
            )
    if not documents:
        raise ValueError("No readable text was found in this PDF.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No readable text chunks could be created from this PDF.")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )
    return vector_db

# -----------------------------
# Display previous chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# Chat input + PDF upload
# -----------------------------
prompt = st.chat_input(
    "Upload a PDF or ask something...",
    accept_file=True,
    file_type=["pdf"],
    max_upload_size=200
)

# -----------------------------
# Handle PDF upload
# -----------------------------
if prompt and prompt.files:
    pdf = prompt.files[0]
    try:
        with st.spinner("Reading your PDF..."):
            st.session_state.vector_db = process_pdf(pdf)
        st.success("PDF uploaded successfully!")
    except ValueError as e:
        st.error(str(e))
    except Exception:
        st.error("Something went wrong while processing the PDF.")

# -----------------------------
# Get user's question
# -----------------------------
question = prompt.text.strip() if prompt else ""

# -----------------------------
# Answer question
# -----------------------------
if question:
    if "vector_db" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        # Save user question to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        # Show user question
        with st.chat_message("user"):
            st.write(question)
        # Search relevant chunks
        try:
            docs = st.session_state.vector_db.similarity_search(
                question,
                k=3
            )
        except Exception:
            docs = []
        if not docs:
            answer = "I couldn't find relevant information in the document."
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
        else:
            # Combine retrieved text with page numbers
            context = "\n\n".join(
                f"Page {doc.metadata.get('page', 'Unknown')}:\n"
                f"{doc.page_content}"
                for doc in docs
            )
            # Prompt for LLM
            prompt_text = f"""
You are ContextAI, a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context.

Follow these rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the document, say:
   "I couldn't find that in the document."
4. Give a clear and well-structured answer.
5. Use headings or bullet points when they make the answer easier to read.
6. Answer directly without unnecessary explanation.
7. Do not mention the document context or retrieval process.

Document context:

{context}

Question:

{question}
"""
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = get_llm().invoke(prompt_text)
                        if response.content:
                            answer = response.content
                            st.write(answer)
                        else:
                            answer = "I couldn't generate an answer."
                            st.write(answer)
                    except Exception:
                        answer = "The model could not generate a response."
                        st.error(answer)
            # Save assistant answer to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })