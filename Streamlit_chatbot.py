import openai
import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
import json
from datetime import datetime

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def save_to_history(question, answer):
    """Save Q&A to session state chat history"""
    st.session_state.chat_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': str(answer)
    })

# Page configuration
st.set_page_config(
    page_title="Climate Change Research Assistant",
    page_icon="🌍",
    layout="wide"
)

# Connecting OpenAI API
openai.api_key = st.secrets["api_secret"]

# Create two columns: main content and chat history
main_col, history_col = st.columns([2, 1])

with main_col:
    # Title and description
    st.title("Climate Change Research Assistant")
    st.markdown("""
    Ask questions about five specific scientific papers on climate change. 
    Your questions will be answered using the knowledge from these papers.
    """)

    # Question input
    query = st.text_input("What would you like to ask?", "")

    # Submit button with error handling
    if st.button("Submit"):
        if not query.strip():
            st.error("Please provide a search query.")
        else:
            try:
                # Show loading spinner
                with st.spinner('Processing your question...'):
                    # Connecting large language model
                    Settings.llm = OpenAI(temperature=0.2, model="gpt-4-1106-preview")
                    
                    # Loading and indexing data
                    documents = SimpleDirectoryReader('./data').load_data()
                    index = VectorStoreIndex.from_documents(documents)
                    
                    # Generating answer
                    query_engine = index.as_query_engine()
                    response = query_engine.query(query)

                    # Display the response in a nice format
                    st.markdown("### Answer:")
                    st.markdown(f">{response}")
                    
                    # Save to chat history
                    save_to_history(query, response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with history_col:
    st.markdown("### Chat History")
    
    # Add a clear history button
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        for i, qa in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {qa['question'][:50]}...", expanded=(i == 0)):
                st.markdown(f"**Time:** {qa['timestamp']}")
                st.markdown(f"**Question:** {qa['question']}")
                st.markdown(f"**Answer:** {qa['answer']}")
    else:
        st.info("No questions asked yet. Try asking something!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Using GPT-4 and LlamaIndex</p>
</div>
""", unsafe_allow_html=True)
