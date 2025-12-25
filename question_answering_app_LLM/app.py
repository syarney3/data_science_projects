import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def load_answer(question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke(question).content

# App UI
st.set_page_config(page_title="LangChain Example", page_icon="🤖")
st.header("OpenAI LangChain Example")

user_input = st.text_input("You:")

if st.button("Generate Answer"):
    if user_input:
        st.subheader("Answer:")
        response = load_answer(user_input)
        st.write(response)
