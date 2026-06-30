import os
import uuid
import hashlib
import streamlit as st

from dotenv import load_dotenv

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY"),
)

if not GROQ_API_KEY:
    st.error("Groq API Key not found.")
    st.stop()


# ============================================================
# Streamlit Page Config
# ============================================================

st.set_page_config(
    page_title="Conversational PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Conversational RAG with PDF")
st.write("Upload one or more PDFs and ask questions about them.")


# ============================================================
# Cache Resources
# ============================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
    )


embeddings = load_embeddings()
llm = load_llm()


# ============================================================
# Session Management
# ============================================================

session_id = st.text_input(
    "Session ID",
    value="default_session",
)

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "last_pdf_hash" not in st.session_state:
    st.session_state.last_pdf_hash = None


# ============================================================
# Chat History
# ============================================================

def get_session_history(session_id):

    if session_id not in st.session_state.sessions:

        st.session_state.sessions[session_id] = {
            "history": ChatMessageHistory(),
            "vectorstore": None,
            "pdf_hash": None,
        }

    return st.session_state.sessions[session_id]["history"]


# ============================================================
# New Chat Button
# ============================================================

if st.button("🗑️ New Chat"):

    if session_id in st.session_state.sessions:
        del st.session_state.store[session_id]

    st.success("Conversation cleared.")

    st.rerun()


# ============================================================
# Upload PDFs
# ============================================================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Please upload one or more PDFs.")
    st.stop()


# ============================================================
# Detect New Upload
# ============================================================

pdf_hash = hashlib.md5()

for file in uploaded_files:
    pdf_hash.update(file.name.encode())
    pdf_hash.update(file.getvalue())

current_hash = pdf_hash.hexdigest()


if session_id not in st.session_state.sessions:

    st.session_state.sessions[session_id] = {
        "history": ChatMessageHistory(),
        "vectorstore": None,
        "pdf_hash": None,
    }

elif st.session_state.sessions[session_id]["pdf_hash"] != current_hash:

    st.session_state.sessions[session_id] = {
        "history": ChatMessageHistory(),
        "vectorstore": None,
        "pdf_hash": current_hash,
    }

# ============================================================
# Load PDFs
# ============================================================

documents = []

for uploaded_file in uploaded_files:

    temp_file = f"temp_{uuid.uuid4().hex}.pdf"

    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_file)

    docs = loader.load()

    documents.extend(docs)

    os.remove(temp_file)

if len(documents) == 0:
    st.error("No text could be extracted from the uploaded PDFs.")
    st.stop()

# ============================================================
# Add Metadata
# ============================================================

for doc in documents:

    if "source" in doc.metadata:

        doc.metadata["file_name"] = os.path.basename(
            doc.metadata["source"]
        )

    else:

        doc.metadata["file_name"] = "Unknown"


# ============================================================
# Split Documents
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splits = text_splitter.split_documents(documents)

if len(splits) == 0:

    st.error("No chunks were created from the uploaded PDFs.")

    st.stop()


# ============================================================
# Build Chroma Vector Database
# ============================================================

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
)

if session_id not in st.session_state.sessions:

    st.session_state.sessions[session_id] = {
        "history": ChatMessageHistory(),
        "vectorstore": vectorstore,
        "pdf_hash": current_hash,
    }

else:

    st.session_state.sessions[session_id]["vectorstore"] = vectorstore
    st.session_state.sessions[session_id]["pdf_hash"] = current_hash
    
vectorstore = st.session_state.sessions[session_id]["vectorstore"]

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)


# ============================================================
# Contextual Question Prompt
# ============================================================

contextualize_q_system_prompt = """
Given the chat history and the latest user question,
which may reference previous conversation,
rewrite the latest question into a standalone question.

Only rewrite the question.

Do NOT answer it.

If no rewrite is required,
return the original question.
"""


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            contextualize_q_system_prompt,
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{input}",
        ),
    ]
)


# ============================================================
# History Aware Retriever
# ============================================================

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# ============================================================
# Question Answer Prompt
# ============================================================

system_prompt = """
You are a helpful AI assistant for question-answering.

Answer ONLY from the retrieved context.

Guidelines:

1. If the answer is present in the context, answer completely.

2. If the answer is partially available,
combine the relevant pieces to produce a complete answer.

3. If the answer is not available,
reply:
"I don't know based on the uploaded document."

4. Do not make up information.

5. When information is spread across multiple chunks,
combine them before answering.

Retrieved Context:

{context}
"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{input}",
        ),
    ]
)


# ============================================================
# QA Chain
# ============================================================

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)


# ============================================================
# Retrieval Chain
# ============================================================

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)


# ============================================================
# Conversational RAG Chain
# ============================================================

conversational_rag_chain = RunnableWithMessageHistory(

    rag_chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="chat_history",

    output_messages_key="answer",

)


# ============================================================
# Display Previous Conversation
# ============================================================

history = get_session_history(session_id)

for message in history.messages:

    if message.type == "human":

        with st.chat_message("user"):

            st.markdown(message.content)

    elif message.type == "ai":

        with st.chat_message("assistant"):

            st.markdown(message.content)


# ============================================================
# User Question
# ============================================================

prompt = st.chat_input(
    "Ask a question about your PDFs..."
)

if prompt:

    with st.chat_message("user"):

        st.markdown(prompt)

    response = conversational_rag_chain.invoke(

        {
            "input": prompt,
        },

        config={
            "configurable": {
                "session_id": session_id,
            }
        },
    )

    answer = response["answer"]

# ============================================================
# Show Assistant Response
# ============================================================

    # Display Assistant Answer
    with st.chat_message("assistant"):

        st.markdown(answer)

        # ----------------------------------------
        # Display Source Documents
        # ----------------------------------------

        if "context" in response:

            with st.expander("📚 Source Documents"):

                for i, doc in enumerate(response["context"], start=1):

                    source = doc.metadata.get(
                        "file_name",
                        "Unknown PDF",
                    )

                    page = doc.metadata.get(
                        "page",
                        "Unknown",
                    )

                    st.markdown(
                        f"### Source {i}"
                    )

                    st.write(
                        f"**PDF:** {source}"
                    )

                    st.write(
                        f"**Page:** {page + 1 if isinstance(page, int) else page}"
                    )

                    st.code(
                        doc.page_content[:800]
                    )

                    st.divider()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📄 Uploaded PDFs")

for file in uploaded_files:

    st.sidebar.success(file.name)

st.sidebar.divider()

st.sidebar.markdown(
"""
### Current Session

Conversation memory is maintained
only for this Session ID.

Changing the Session ID creates
a new conversation.

Uploading different PDFs clears
the previous conversation automatically.
"""
)

st.sidebar.divider()

st.sidebar.markdown(
"""
### Model

- LLM : llama-3.3-70b-versatile
- Embedding :
sentence-transformers/all-MiniLM-L6-v2
- Vector DB : Chroma
"""
)

st.sidebar.divider()

st.sidebar.markdown(
"""
### Retrieval

- Conversational RAG
- History Aware Retriever
- Top-k Retrieval
"""
)
