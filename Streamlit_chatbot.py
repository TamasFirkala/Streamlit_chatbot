import streamlit as st
import os
import json

st.title("Pinecone API Test")

# Get API keys from Streamlit secrets
pinecone_api_key = st.secrets.get("pinecone_api_key", "Not found in secrets")
pinecone_env = st.secrets.get("pinecone_environment", "Not found in secrets")
pinecone_index_name = st.secrets.get("pinecone_index_name", "Not found in secrets")

# Display masked API key info for verification
st.write(f"API Key available: {'Yes' if pinecone_api_key != 'Not found in secrets' else 'No'}")
st.write(f"Environment: {pinecone_env}")
st.write(f"Index Name: {pinecone_index_name}")

# Try both Pinecone V1 and V2 approaches
st.subheader("Testing Pinecone Connection")

try:
    # Try V2 first
    st.write("Attempting V2 connection...")
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    index = pc.Index(pinecone_index_name)
    
    # Test if we can get index stats
    stats = index.describe_index_stats()
    st.success("✅ Pinecone V2 connection successful!")
    
    # Convert to dict first to ensure valid JSON
    stats_dict = vars(stats) if hasattr(stats, '__dict__') else stats
    if isinstance(stats_dict, dict):
        st.write("Index Statistics:")
        st.write(stats_dict)
    else:
        st.write("Raw Statistics:", stats)
    
except ImportError:
    st.warning("V2 import failed, trying V1...")
    try:
        import pinecone
        
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        index = pinecone.Index(pinecone_index_name)
        
        # Test if we can get index stats
        stats = index.describe_index_stats()
        st.success("✅ Pinecone V1 connection successful!")
        
        if isinstance(stats, dict):
            st.write("Index Statistics:")
            st.write(stats)
        else:
            st.write("Raw Statistics:", stats)
        
    except Exception as e:
        st.error(f"❌ Pinecone V1 connection failed: {str(e)}")
        
except Exception as e:
    st.error(f"❌ Pinecone V2 connection failed: {str(e)}")

# Display package versions for debugging
st.subheader("Package Versions")
import pkg_resources

for pkg in pkg_resources.working_set:
    if any(x in pkg.key for x in ["pinecone", "llama", "openai"]):
        st.write(f"- {pkg.key}: {pkg.version}")
