import streamlit as st
import pinecone  # Changed from 'from pinecone import Pinecone'
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai

# Page configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="💬"
)

# Initialize all components together
@st.cache_resource
def init_components():
    # Set OpenAI API key
    openai.api_key = st.secrets["openai_api_key"]
    
    # Initialize LLM
    llm = OpenAI(
        api_key=st.secrets["openai_api_key"],
        temperature=0.2,
        model="gpt-4-1106-preview"
    )
    
    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        api_key=st.secrets["openai_api_key"],
        model_name="text-embedding-3-small",
        dimensions=384
    )
    
    # Initialize Pinecone
    pinecone.init(
        api_key=st.secrets["pinecone_api_key"]
    )
    
    # Get the index
    pinecone_index = pinecone.Index(st.secrets["pinecone_index_name"])
    
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
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    return llm, embed_model, query_engine

def main():
    st.title("💬 Document Q&A")
    
    # Display current configuration
    with st.expander("Configuration Status"):
        st.write("Checking configuration...")
        if "openai_api_key" in st.secrets:
            st.write("✅ OpenAI API Key configured")
        if "pinecone_api_key" in st.secrets:
            st.write("✅ Pinecone API Key configured")
        if "pinecone_index_name" in st.secrets:
            st.write(f"📍 Pinecone Index: {st.secrets['pinecone_index_name']}")
    
    try:
        # Initialize all components
        with st.spinner("Initializing components..."):
            llm, embed_model, query_engine = init_components()
            st.success("All components initialized successfully!")
        
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
        st.error("Please check your API keys and configurations.")

if __name__ == "__main__":
    main()
