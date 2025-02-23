import streamlit as st
import openai
from pinecone import Pinecone as PineconeClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=st.secrets["openai_api_key"])
Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-small",
    dimensions=384,
    api_key=st.secrets["openai_api_key"]
)

st.title("Query Test")

# Global variables to store the query engine
query_engine = None

try:
    # Initialize components
    embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        dimensions=384,
        api_key=st.secrets["openai_api_key"]
    )
    
    # Initialize OpenAI LLM
    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        api_key=st.secrets["openai_api_key"]
    )
    
    # Update Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    pc = PineconeClient(
        api_key=st.secrets["pinecone_api_key"],
        environment=st.secrets["pinecone_environment"]
    )
    index_name = st.secrets["pinecone_index_name"]
    pinecone_index = pc.Index(index_name)
    
    # Create vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )
    
    # Create vector store index
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store
    )
    
    # Create query engine
    query_engine = vector_index.as_query_engine()

    # Query interface
    st.markdown("### Test Query Interface")
    test_query = st.text_input("Enter your query:", "What is this document about?")
    if st.button("Run Query", key="query_button"):
        st.write("Button clicked!")  # Debug message
        try:
            st.write("Executing query...")  # Debug message
            with st.spinner('Processing query...'):
                response = query_engine.query(test_query)
                st.write("Query completed!")  # Debug message
                st.write("Response:", str(response))
        except Exception as e:
            st.error(f"Query Error: {str(e)}")

except Exception as e:
    st.error(f"Setup Error: {str(e)}")
