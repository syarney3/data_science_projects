import streamlit as st
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

from dotenv import load_dotenv
load_dotenv()


#By using st.set_page_config(), you can customize the appearance of your Streamlit application's web page
st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

#Initialize the OpenAIEmbeddings object
embeddings = OpenAIEmbeddings()

#import CSV file
loader = CSVLoader(file_path='similar_words_finder_app_LLM\myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})
data = loader.load()

db = FAISS.from_documents(data, embeddings)

# Function to receive input from user and store it in a variable
def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text


user_input=get_text()
submit = st.button('Find similar Things')  

if submit:
    
    #If the button is clicked, the below snippet will fetch us the similar text
    docs = db.similarity_search(user_input)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)
