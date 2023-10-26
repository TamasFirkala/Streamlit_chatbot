import os
import openai

#os.environ['OPENAI_API_KEY'] = 'API_KEY'

#openai.api_key  = 'API_KEY'

import streamlit as st

# Create an index of your documents

st.title("Ask Llama")

query = st.text_input("What would you like to ask? (source: /data )", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:

            from llama_index import SimpleDirectoryReader

            openai.api_key = os.environ["API_KEY"]

            documents = SimpleDirectoryReader('./data').load_data() 

            from llama_index import GPTVectorStoreIndex

            index = GPTVectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()

            response = query_engine.query(query)



            
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")


