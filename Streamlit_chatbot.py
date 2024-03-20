
import openai
import os
import streamlit as st

#Connecting OpenAI API
openai.api_key = st.secrets["api_secret"]

#Frontend preparation uisng Python-streamlit package

#title
st.title("You can ask ChatGPT about your own data. It learned the next five specific scientific papers in the topic of climate change. Don't hesitate to ask about them!")

#question
query = st.text_input("What would you like to ask?", "")

#"submit" button
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:

            #Connecting large language model using Python Llama_index package
            
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
            from llama_index.llms.openai import OpenAI

            #Choosing large language model
            Settings.llm = OpenAI(temperature=0.2, model="gpt-4-1106-preview")
            
            #loading data (the articles about climate change)
            documents = SimpleDirectoryReader('./data').load_data()

            #indexing data
            index = VectorStoreIndex.from_documents(documents)
            
            #generating answer
            
            query_engine = index.as_query_engine()
            response = query_engine.query(query)

            
                      
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")




