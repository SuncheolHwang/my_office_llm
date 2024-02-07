import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOllama
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache("cache.db"))

st.set_page_config(
    page_title="Llamma",
    page_icon="📃",
)

callback = False


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        if callback:
            self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        if callback:
            save_messages(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        if callback:
            self.message += token
            self.message_box.markdown(self.message)


if "llamma_messages" not in st.session_state:
    st.session_state["llamma_messages"] = []

if "prompt_code" not in st.session_state:
    st.session_state["prompt_code"] = "C language"

prompt_message_mistral = "[INST]You are an engineering expert. explain your question in detail. And provide additional definitions for technical terms.[/INST]"
prompt_message_llammachat = "You are an engineering expert. explain your question in detail. And provide additional definitions for technical terms."
prompt_translate="You are an engineer translation expert. Translate the following sentence naturally into Korean."

def prompt_message_codellama(language):
    return f"You are an expert programmer that helps to write {language} code based on the user request, with concise explanations. Don't be too verbose."

def select_model(model):
    if (model == "Mistral"):
        return "mistral:latest"
    elif (model == "Code Llama"):
        return "codellama:34b"
    elif (model == "Llamma Chat"):
        return "llama2:13b-chat"

def set_prompt(model):
    if (model == "Mistral"):
        return prompt_message_mistral
    elif (model == "Code Llama"):
        return prompt_message_codellama(st.session_state["prompt_code"])
    else:
        return prompt_message_llammachat


with st.sidebar:
    choice = st.selectbox("Select Model", ["Mistral",  "Code Llama", "Llamma Chat"])
    if choice == "Code Llama":
        code = st.selectbox("Select Language", ["C++ language", "C language", "C++ high level synthesis", "python"])
        st.session_state["prompt_code"] = code

    prompt_text = st.text_area(
        "Prompt", set_prompt(choice),
    )

    choice_model = select_model(choice)


llm = ChatOllama(
    temperature=0.1,
    model=choice_model,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
    return_messages=True,
)

if "llamma_chat_summary" not in st.session_state:
    st.session_state["llamma_chat_summary"] = []
    st.session_state["last_answer"] = ""
else:
    callback = False
    if st.session_state["llamma_chat_summary"]:
        st.session_state["last_answer"] = st.session_state["llamma_chat_summary"][-1]["answer"]

    for chat_list in st.session_state["llamma_chat_summary"]:
        memory.save_context(
            {"input": chat_list["question"]},
            {"output": chat_list["answer"]},
        )


def save_messages(message, role):
    st.session_state["llamma_messages"].append(
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
    for message in st.session_state["llamma_messages"]:
        send_message(message["message"], message["role"], save=False)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {prompt_text}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

def load_memory(_):
    loaded_memory = memory.load_memory_variables({})["history"]
    return loaded_memory


def save_context(question, result):
    st.session_state["llamma_chat_summary"].append(
        {
            "question": question,
            "answer": result,
        }
    )

def invoke_chain(question):
    result = chain.invoke(
        {"question": question},
    )
    save_context(message, result.content)

@st.spinner(text="translating...")
def translate_answer():
    sentence = st.session_state["last_answer"]
    print(sentence)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                {prompt_translate}
                """,
            ),
            ("human", "{question}"),
        ]
    )
    translate_llm = ChatOllama(
        temperature=0.1,
        model="mistral:latest",
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    
    chain = prompt | translate_llm
    chain.invoke(
        {"question": sentence}
    )

st.title("Llamma Chatbot")

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
        chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm

        with st.chat_message("ai"):
            callback = True
            invoke_chain(message)
    
    if st.button("translate"):
        with st.chat_message("ai"):
            callback = True
            translate_answer()

