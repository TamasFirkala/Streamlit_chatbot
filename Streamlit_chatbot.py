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
            pc = PineconeClient(
                api_key=st.secrets["pinecone_api_key"],
                environment=st.secrets["pinecone_environment"]
            )

            # Try to get index information
            index_name = st.secrets["pinecone_index_name"]
            index = pc.Index(index_name)
            index_stats = index.describe_index_stats()

            st.success(f"""
                Pinecone connection successful!
                - API key is valid
                - Index '{index_name}' exists
                - Total vectors: {index_stats.total_vector_count}
                - Dimension: {index_stats.dimension}
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
            
            # Initialize OpenAI LLM
            llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=st.secrets["openai_api_key"]
            )
            
            # Update Settings instead of using ServiceContext
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
            
            # Create vector store index without service_context
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store
            )
            
            # Simple test query to verify connection
            query_engine = vector_index.as_query_engine()
            stats = pinecone_index.describe_index_stats()
            
            st.success(f"""
                LlamaIndex-Pinecone connection successful!
                - Vector store connected
                - Number of vectors in index: {stats.total_vector_count}
                - Dimension: {stats.dimension}
                
                Try a test query below:
            """)
            
            # Add a simple query interface
            test_query = st.text_input("Enter a test query:", "What is this document about?")
            if st.button("Run Query"):
                with st.spinner("Generating response..."):
                    try:
                        response = query_engine.query(test_query)
                        st.write("Raw Response Object:", type(response))  # Debug info
                        st.write("Response Content:", str(response))
                        
                        # Try accessing different response attributes
                        st.write("Response as string:", str(response))
                        if hasattr(response, 'response'):
                            st.write("Response.response:", response.response)
                        if hasattr(response, 'text'):
                            st.write("Response.text:", response.text)
                        
                        # Display source nodes if available
                        if hasattr(response, 'source_nodes'):
                            st.write("Source Nodes:")
                            for node in response.source_nodes:
                                st.write("- Source:", node.node.text[:200] + "...")
                                
                    except Exception as e:
                        st.error(f"Error processing response: {str(e)}")
                        st.write("Full error:", e)

    except Exception as e:
        st.error(f"LlamaIndex-Pinecone Integration Error: {str(e)}")

st.markdown("---")
st.markdown("""
### How to use:
1. Click each test button to verify the corresponding API connection
2. Check the configured secrets to ensure all required keys are present
3. If you see any errors, verify your API keys and index name in the Streamlit secrets
""")
