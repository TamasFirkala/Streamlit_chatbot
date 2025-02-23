import streamlit as st
import pinecone  # Changed from 'from pinecone import Pinecone'
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="💬"
)

# Initialize embedding model
@st.cache_resource
def init_embeddings():
    return OpenAIEmbedding(
        model_name="text-embedding-3-small",
        dimensions=384,
        api_key=st.secrets["openai_api_key"]
    )

# Initialize LLM
@st.cache_resource
def init_llm():
    return OpenAI(
        temperature=0.2, 
        model="gpt-4-1106-preview", 
        api_key=st.secrets["openai_api_key"]
    )

# Initialize Pinecone and create query engine
@st.cache_resource
def init_query_engine():
    # Initialize Pinecone
    pinecone.init(
        api_key=st.secrets["pinecone_api_key"],
        environment=st.secrets["pinecone_environment"]  # Make sure to add this to your secrets
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
        embed_model=init_embeddings()
    )
    
    return index.as_query_engine()

# Main app
def main():
    st.title("💬 Document Q&A")
    
    # Initialize components
    embed_model = init_embeddings()
    llm = init_llm()
    query_engine = init_query_engine()
    
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
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
