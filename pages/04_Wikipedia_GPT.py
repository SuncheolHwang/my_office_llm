from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough

st.set_page_config(
    page_title="Wikipedia GPT",
    page_icon="❓",
)

st.title("Wikipedia GPT")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_messages(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


if "wiki_messages" not in st.session_state:
    st.session_state["wiki_messages"] = []

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context, tell me about {word} in ONLY Korean.
    The content should be summarized in 20 lines or less, focusing on the technical aspects and using the most up-to-date data if possible.
    And it doesn't have to be up-to-date if you think it's important.
    If it's the same word but has a different meaning, add "----------"
    
    Context: {context}
""",
        )
    ]
)

docs = None


def format_docs(topic):
    return {
        "word": topic,
        "context": "\n\n".join(document.page_content for document in docs),
    }


chain = format_docs | prompt | llm


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


def save_messages(message, role):
    st.session_state["wiki_messages"].append(
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
    for message in st.session_state["wiki_messages"]:
        send_message(message["message"], message["role"], save=False)


send_message("Search from Wikidepia...", "ai", save=False)
paint_history()

authentication_status = st.session_state["authentication_status"]
if authentication_status:
    message = st.chat_input("Enter the word you want to know....")

    if message:
        send_message(message, "human")
        with st.chat_message("ai"):
            docs = wiki_search(message)
            if docs:
                chain.invoke(message)
