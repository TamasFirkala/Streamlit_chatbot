
import openai
import os
import streamlit as st

openai.api_key = st.secrets["api_secret"]

st.title("Ask ChatGPT about your own data!")

query = st.text_input("What would you like to ask?", "")

if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:

            from llama_index import SimpleDirectoryReader

            documents = SimpleDirectoryReader('./data').load_data()

            from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper
            from langchain.llms import OpenAI

            #llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="gpt-4-1106-preview"))
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-003"))

            max_input_size = 4096
            num_output = 256
            max_chunk_overlap = 0.5
            prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

            custom_LLM_index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper = prompt_helper)

            query_engine = custom_LLM_index.as_query_engine()

            response = query_engine.query(query)

            

            """from llama_index import GPTVectorStoreIndex

            index = GPTVectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()

            response = query_engine.query(query)"""



          
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")




