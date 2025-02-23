import streamlit as st
import openai
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding

st.title("API Connection Test")

# Add expanders for each test
with st.expander("Test OpenAI Connection"):
    try:
        # Test OpenAI connection
        if st.button("Test OpenAI API"):
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            
            # Try to create a simple embedding
            embed_model = OpenAIEmbedding(
                model_name="text-embedding-3-small",
                dimensions=384,
                api_key=st.secrets["OPENAI_API_KEY"]
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
            pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
            
            # Try to get index information
            index_name = st.secrets["PINECONE_INDEX_NAME"]
            index_info = pc.describe_index(index_name)
            
            st.success(f"""
                Pinecone connection successful!
                - API key is valid
                - Index '{index_name}' exists
                - Index dimension: {index_info.dimension}
                - Index metric: {index_info.metric}
                - Index pods: {index_info.pod_type}
            """)
            
    except Exception as e:
        st.error(f"Pinecone API Error: {str(e)}")

# Display current secrets (without showing actual values)
with st.expander("Check Configured Secrets"):
    st.write("Checking for required secrets...")
    
    # Check if each required secret exists
    secrets_status = {
        "OPENAI_API_KEY": "OPENAI_API_KEY" in st.secrets,
        "PINECONE_API_KEY": "PINECONE_API_KEY" in st.secrets,
        "PINECONE_INDEX_NAME": "PINECONE_INDEX_NAME" in st.secrets
    }
    
    for secret_name, exists in secrets_status.items():
        if exists:
            st.success(f"✅ {secret_name} is configured")
        else:
            st.error(f"❌ {secret_name} is missing")

st.markdown("---")
st.markdown("""
### How to use:
1. Click each test button to verify the corresponding API connection
2. Check the configured secrets to ensure all required keys are present
3. If you see any errors, verify your API keys and index name in the Streamlit secrets
""")
