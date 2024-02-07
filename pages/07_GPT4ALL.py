import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import GPT4AllEmbeddings
import os
from langchain.llms.gpt4all import GPT4All
from langchain.embeddings import OpenAIEmbeddings

st.set_page_config(
    page_title="GPT4All Chatbot with document",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_messages(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 100  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = GPT4All(
    model="./gpt4all-falcon-q4_0.gguf",
    max_tokens=3000,
    device="gpu",
    top_p=1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
    verbose=True,  # Verbose is required to pass to the callback manager
    use_mlock=False,
    n_batch=n_batch,
)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/gpt4files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/gpt4embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = Chroma.from_documents(docs, cached_embeddings)
    retriver = vectorstore.as_retriever()
    return retriver


def save_messages(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_messages(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using the following context.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("GPT4All Chatbot with Document")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on sidebar
    """
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
else:
    st.session_state["messages"] = []
