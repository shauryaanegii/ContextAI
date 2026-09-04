import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="ContextAI",
    page_icon="📕",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

.welcome {
    text-align: center;
    padding: 150px 20px 50px;
}

.welcome-icon {
    font-size: 45px;
    margin-bottom: 18px;
}

.welcome-title {
    font-size: 32px;
    font-weight: 750;
    margin-bottom: 10px;
}

.welcome-text {
    max-width: 620px;
    margin: auto;
    color: #858994;
    font-size: 15px;
    line-height: 1.6;
}

[data-testid="stChatInput"] {
    border-radius: 16px;
}

.stChatMessage {
    border-radius: 15px;
}

div.stButton > button {
    border-radius: 10px;
}

div[data-testid="stDownloadButton"] button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2
    )


def create_retriever(file):
    reader = PdfReader(file)
    documents = []

    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text()

        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file.name,
                        "page": page_number
                    }
                )
            )

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


def ask_document(question):
    docs = st.session_state.retriever.invoke(question)

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Document')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are ContextAI, an intelligent document assistant.

Use only the information provided in the document context to answer the question.

If the answer is not present in the context, say:
"I couldn't find that in the document."

Do not invent information.

Give clear, natural and easy-to-understand answers.

You can explain, summarize, compare or clarify information
when supported by the document."""
        ),
        (
            "human",
            "Document context:\n\n{context}\n\nQuestion:\n\n{question}"
        )
    ])

    response = get_llm().invoke(
        prompt.invoke({
            "context": context,
            "question": question
        })
    )

    return response.content


chat = st.chat_input(
    "Ask anything about your document..."
    if st.session_state.document_name
    else "Attach a PDF to start chatting...",
    accept_file=True,
    file_type=["pdf"],
    max_upload_size=200
)


if chat:
    files = chat.files
    question = chat.text.strip()

    if files:
        file = files[0]

        if file.name != st.session_state.document_name:
            with st.spinner("Reading your document..."):
                try:
                    retriever = create_retriever(file)

                    if retriever:
                        st.session_state.retriever = retriever
                        st.session_state.document_name = file.name
                        st.session_state.messages = []

                    else:
                        st.error(
                            "I couldn't read text from this document."
                        )

                except Exception:
                    st.error(
                        "I couldn't open this document. "
                        "Please try another PDF."
                    )

    if question and st.session_state.retriever:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_document(question)
                except Exception:
                    answer = (
                        "I couldn't complete that request right now. "
                        "Please try again."
                    )

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    elif question and not st.session_state.retriever:
        st.info("Attach a PDF first to start chatting.")


if not st.session_state.document_name:
    st.markdown(
        """<div class="welcome">
        <div class="welcome-icon">✦</div>
        <div class="welcome-title">Ask and Understand</div>
        <div class="welcome-text">
        Attach a PDF using the button in the chat bar below
        and start a conversation with its content.
        </div>
        </div>""",
        unsafe_allow_html=True
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.document_name:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("＋ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("↻ New Document", use_container_width=True):
            st.session_state.messages = []
            st.session_state.retriever = None
            st.session_state.document_name = None
            st.rerun()


if st.session_state.messages:
    chat_text = "\n\n".join(
        (
            "You:"
            if message["role"] == "user"
            else "ContextAI:"
        )
        + "\n"
        + message["content"]
        for message in st.session_state.messages
    )

    st.download_button(
        "↓ Download Chat",
        data=chat_text,
        file_name="contextai-chat.txt",
        mime="text/plain",
        use_container_width=True
    )