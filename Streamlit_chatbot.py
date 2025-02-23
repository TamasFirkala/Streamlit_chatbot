import streamlit as st
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai  # Add this import

# Page configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="💬"
)

# Check for required secrets
def check_secrets():
    required_secrets = [
        "openai_api_key",
        "pinecone_api_key",
        "pinecone_index_name"
    ]
    
    missing_secrets = [secret for secret in required_secrets if secret not in st.secrets]
    
    if missing_secrets:
        st.error(f"Missing required secrets: {', '.join(missing_secrets)}")
        st.write("Please add these secrets in your Streamlit configuration.")
        st.stop()

# Initialize OpenAI
@st.cache_resource
def init_openai():
    openai.api_key = st.secrets["openai_api_key"]
    
    return OpenAI(
        api_key=st.secrets["openai_api_key"],
        temperature=0.2,
        model="gpt-4-1106-preview"
    )

# Initialize embedding model
@st.cache_resource
def init_embeddings():
    return OpenAIEmbedding(
        api_key=st.secrets["openai_api_key"],
        model_name="text-embedding-3-small",
        dimensions=384
    )

# Initialize Pinecone and create query engine
@st.cache_resource
def init_query_engine(embed_model):
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
        
        # Get the index
        pinecone_index = pc.Index(st.secrets["pinecone_index_name"])
        
        # Create vector store
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            dimension=384
        )
        
        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        
        return index.as_query_engine()
    
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        st.write("Please check your Pinecone configuration.")
        raise e

# Main app
def main():
    st.title("💬 Document Q&A")
    
    # Check for required secrets first
    check_secrets()
    
    # Display current configuration (without showing actual keys)
    with st.expander("Configuration Status"):
        st.write("✅ OpenAI API Key configured")
        st.write("✅ Pinecone API Key configured")
        st.write(f"📍 Pinecone Index: {st.secrets['pinecone_index_name']}")
    
    try:
        # Initialize OpenAI first
        llm = init_openai()
        st.write("OpenAI initialized successfully")
        
        # Initialize embeddings
        embed_model = init_embeddings()
        st.write("Embeddings initialized successfully")
        
        # Initialize query engine with embeddings
        query_engine = init_query_engine(embed_model)
        st.write("Query engine initialized successfully")
        
        # Create the question input
        question = st.text_input("Ask a question about your documents:")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        # Get response from query engine
                        response = query_engine.query(question)
                        
                        # Display response
                        st.markdown("### Answer:")
                        st.write(response)
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")

if __name__ == "__main__":
    main()
