import os
import uuid
import streamlit as st

from dotenv import load_dotenv

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================================================
# Load Environment Variables
# ==================================================

load_dotenv()

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Configure it in .env or Streamlit Secrets.")
    st.stop()

# ==================================================
# Streamlit Page Config
# ==================================================

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Conversational RAG with PDF")
st.write("Upload one or more PDFs and chat with their content.")

# ==================================================
# Session Management
# ==================================================

session_id = st.text_input(
    "Session ID",
    value="default_session"
)

if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# Embeddings
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# LLM
# ==================================================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    # model_name="llama-3.3-70b-versatile"
    model_name="deepseek-r1-distill-llama-70b"
)

# ==================================================
# Chat History Function
# ==================================================

def get_session_history(
    session: str
) -> BaseChatMessageHistory:

    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()

    return st.session_state.store[session]

# ==================================================
# PDF Upload
# ==================================================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    documents = []

    for uploaded_file in uploaded_files:

        temp_file = f"temp_{uuid.uuid4().hex}.pdf"

        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_file)
        docs = loader.load()

        documents.extend(docs)

        os.remove(temp_file)

    # ==============================================
    # Split Documents
    # ==============================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)

    if len(splits) == 0:
        st.error(
            "No text could be extracted from the uploaded PDF(s)."
        )
        st.stop()

    # ==============================================
    # Create Vector Store
    # ==============================================

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)

    # ==============================================
    # Contextual Question Reformulation
    # ==============================================

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be "
        "understood without the chat history. "
        "Do NOT answer the question."
        "Answer to the question only if it present in chat history or provided context"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # ==============================================
    # QA Prompt
    # ==============================================

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If you do not know the answer, say that you do not know. "
        "Keep the answer concise.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # ==============================================
    # Display Existing Messages
    # ==============================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ==============================================
    # User Input
    # ==============================================

    if prompt := st.chat_input(
        "Ask a question about the uploaded PDFs..."
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        response = conversational_rag_chain.invoke(
            {"input": prompt},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

else:
    st.info("Please upload at least one PDF to start chatting.")
