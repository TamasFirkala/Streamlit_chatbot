
import openai
import os
import streamlit as st

openai.api_key = st.secrets["api_secret"]

st.title("Hi I am Bela, a Hungarian chatbot. I am presently being trained to help you with chemistry. Don't hesitate to ask me!")

query = st.text_input("What would you like to ask?", "")

if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:

            from llama_index import SimpleDirectoryReader

            documents = SimpleDirectoryReader('./data').load_data()

            from llama_index import GPTVectorStoreIndex

            index = GPTVectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()

            response = query_engine.query(query)



          
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")




