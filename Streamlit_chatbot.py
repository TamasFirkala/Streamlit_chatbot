import streamlit as st
import openai
import pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.types import VectorStore
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.query.query_transform.contracts import QueryTransform
from llama_index.indices.query.visualizations import QueryVisualization

# Initialize OpenAI settings
openai.api_key = st.secrets["openai_api_key"]

# Initialize session state variables if not already set
if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = None

st.title("API Connection Test")

# Add expanders for each test
with st.expander("Test LlamaIndex-Pinecone Connection"):
    if st.button("Initialize LlamaIndex-Pinecone Integration"):
        try:
            # Initialize components
            embed_model = OpenAIEmbedding(
                model_name="text-embedding-ada-002",
                api_key=st.secrets["openai_api_key"]
            )
            
            # Initialize OpenAI LLM
            llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=st.secrets["openai_api_key"]
            )
            
            # Create a service context
            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embed_model
            )
            
            # Initialize Pinecone
            pinecone.init(
                api_key=st.secrets["pinecone_api_key"],
                environment=st.secrets["pinecone_environment"]
            )
            index_name = st.secrets["pinecone_index_name"]
            pinecone_index = pinecone.Index(index_name)
            
            # Create vector store
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index
            )
            
            # Create vector store index without service_context
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store,
                service_context=service_context
            )
            
            # Create a query engine and store in session state
            st.session_state['query_engine'] = vector_index.as_query_engine()
            
            # Success message
            st.success("LlamaIndex-Pinecone integration initialized successfully!")
        
        except Exception as e:
            st.error(f"Initialization Error: {str(e)}")

# Query Interface
st.header("Run a Query")

# Check if the query engine is initialized
if st.session_state['query_engine'] is None:
    st.warning("Please initialize the LlamaIndex-Pinecone integration first.")
else:
    # Create the query interface
    test_query = st.text_input("Enter a test query:", "What is this document about?")
    if st.button("Run Query"):
        try:
            response = st.session_state['query_engine'].query(test_query)
            st.write("**Response:**")
            st.write(response.response)
        except Exception as e:
            st.error(f"Error during query execution: {str(e)}")
