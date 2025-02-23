import streamlit as st
import openai
from pinecone import Pinecone as PineconeClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext

st.title("API Connection Test")

# Add expanders for each test
with st.expander("Test OpenAI Connection"):
    try:
        # Test OpenAI connection
        if st.button("Test OpenAI API"):
            openai.api_key = st.secrets["openai_api_key"]

            # Try to create a simple embedding
            embed_model = OpenAIEmbedding(
                model_name="text-embedding-3-small",
                dimensions=384,
                api_key=st.secrets["openai_api_key"]
            )
             
            test_embedding = embed_model.get_text_embedding("Hello, world!")

            st.success(f"""
                OpenAI connection successful!
                - API key is valid
                - Embedding model working
                - Embedding dimension: {len(test_embedding)}
            """)

    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")

with st.expander("Test Pinecone Connection"):
    try:
        # Test Pinecone connection
        if st.button("Test Pinecone API"):
            # Initialize Pinecone client
            pc = PineconeClient(
                api_key=st.secrets["pinecone_api_key"],
                environment=st.secrets["pinecone_environment"]
            )

            # Try to get index information
            index_name = st.secrets["pinecone_index_name"]
            index_info = pc.describe_index(index_name)

            st.success(f"""
                Pinecone connection successful!
                - API key is valid
                - Index '{index_name}' exists
                - Dimension: {index_info.dimension}
                - Metric: {index_info.metric}
            """)

    except Exception as e:
        st.error(f"Pinecone API Error: {str(e)}")

# Display current secrets (without showing actual values)
with st.expander("Check Configured Secrets"):
    st.write("Checking for required secrets...")
    
    # Check if each required secret exists
    secrets_status = {
        "OPENAI_API_KEY": "openai_api_key" in st.secrets,
        "PINECONE_API_KEY": "pinecone_api_key" in st.secrets,
        "PINECONE_INDEX_NAME": "pinecone_index_name" in st.secrets,
        "PINECONE_ENVIRONMENT": "pinecone_environment" in st.secrets
    }
    
    for secret_name, exists in secrets_status.items():
        if exists:
            st.success(f"✅ {secret_name} is configured")
        else:
            st.error(f"❌ {secret_name} is missing")

with st.expander("Test LlamaIndex-Pinecone Connection"):
    try:
        if st.button("Test LlamaIndex-Pinecone Integration"):
            # Initialize components
            embed_model = OpenAIEmbedding(
                model_name="text-embedding-3-small",
                dimensions=384,
                api_key=st.secrets["openai_api_key"]
            )
            
            # Initialize Pinecone client
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
            
            # Create vector store index with embed_model directly
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=embed_model
            )
            
            # Simple test query to verify connection
            query_engine = vector_index.as_query_engine()
            stats = pinecone_index.describe_index_stats()
            
            st.success(f"""
                LlamaIndex-Pinecone connection successful!
                - Vector store connected
                - Index statistics retrieved
                - Connection verified
                
                Try a test query below:
            """)
            
            # Add a simple query interface
            test_query = st.text_input("Enter a test query:", "What is this document about?")
            if st.button("Run Query"):
                response = query_engine.query(test_query)
                st.write("Response:", response)

    except Exception as e:
        st.error(f"LlamaIndex-Pinecone Integration Error: {str(e)}")

st.markdown("---")
st.markdown("""
### How to use:
1. Click each test button to verify the corresponding API connection
2. Check the configured secrets to ensure all required keys are present
3. If you see any errors, verify your API keys and index name in the Streamlit secrets
""")
