
import openai
import os
import streamlit as st

openai.api_key = st.secrets["api_secret"]

st.title("Ask Llama")

query = st.text_input("What would you like to ask? (source: /content/data_source)", "")

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




