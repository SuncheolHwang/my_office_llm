import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory

st.set_page_config(
    page_title="Gemini-Pro",
    page_icon="📃",
)


if "gemini_messages" not in st.session_state:
    st.session_state["gemini_messages"] = []

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
)

memory = ConversationBufferWindowMemory(
    llm=llm,
    k=2,
    return_messages=True,
)

if "gemini_chat_summary" not in st.session_state:
    st.session_state["gemini_chat_summary"] = []
else:
    for chat_list in st.session_state["gemini_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["gemini_messages"].append(
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
    for message in st.session_state["gemini_messages"]:
        send_message(message["message"], message["role"], save=False)


st.title("Gemini-Pro Chatbot")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your Questions!
    
    """
)

with st.sidebar:
    prompt_text = st.text_area(
        "Prompt",
        " -----------------------------------------------\nexplain previous sentences in detail in Korean.",
    )

send_message("I'm ready! Ask away!", "ai", save=False)
paint_history()

authentication_status = st.session_state["authentication_status"]
if authentication_status:
    message = st.chat_input("Ask anything about something...")

    if message:
        send_message(message, "human")
        with st.chat_message("ai"):
            ai_message = ""
            for chunk in llm.stream(message + prompt_text):
                chunk.content
                ai_message += "\n" + chunk.content
            save_messages(ai_message, "ai")
