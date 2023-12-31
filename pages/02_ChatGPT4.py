import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="ChatGPT4",
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


if "gpt4_messages" not in st.session_state:
    st.session_state["gpt4_messages"] = []

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4-1106-preview",
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


def save_messages(message, role):
    st.session_state["gpt4_messages"].append(
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
    for message in st.session_state["gpt4_messages"]:
        send_message(message["message"], message["role"], save=False)


with st.sidebar:
    prompt_text = st.text_area(
        "Prompt",
        """You are an engineering expert. explain the question in detail.""",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {prompt_text}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("ChatGPT4 Chatbot")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your Questions!
    
    """
)

send_message("I'm ready! Ask away!", "ai", save=False)
paint_history()

authentication_status = st.session_state["authentication_status"]
if authentication_status:
    message = st.chat_input("Ask anything about something...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
