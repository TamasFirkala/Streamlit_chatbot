import openai
import os
import streamlit as st
from datetime import datetime
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from pinecone import Pinecone

# Papers Information Dictionary (your existing PAPERS_INFO dictionary)
PAPERS_INFO = {
    # ... (keep all your existing paper information)
}

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'apis_tested' not in st.session_state:
    st.session_state.apis_tested = False

def test_api_connections():
    """Test OpenAI and Pinecone API connections"""
    with st.expander("API Connection Test"):
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                if st.button("Test OpenAI API"):
                    openai.api_key = st.secrets["openai_api_key"]
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

        with col2:
            try:
                if st.button("Test Pinecone API"):
                    pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
                    index_name = st.secrets["pinecone_index_name"]
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

        # Display current secrets status
        with st.expander("Check Configured Secrets"):
            st.write("Checking for required secrets...")
            secrets_status = {
                "openai_api_key": "openai_api_key" in st.secrets,
                "pinecone_api_key": "pinecone_api_key" in st.secrets,
                "pinecone_index_name": "pinecone_index_name" in st.secrets
            }
            for secret_name, exists in secrets_status.items():
                if exists:
                    st.success(f"✅ {secret_name} is configured")
                else:
                    st.error(f"❌ {secret_name} is missing")

def initialize_llama_index():
    """Initialize LlamaIndex with Pinecone"""
    # Initialize OpenAI and Pinecone
    openai.api_key = st.secrets["openai_api_key"]
    
    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        model_name="text-embedding-3-small",
        dimensions=384,
        api_key=st.secrets["openai_api_key"]
    )

    # Initialize LLM
    llm = OpenAI(temperature=0.2, model="gpt-4-1106-preview", api_key=st.secrets["openai_api_key"])

    # Update global settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.node_parser = SimpleNodeParser.from_defaults()

    # Initialize Pinecone
    pc = Pinecone(api_key=st.secrets["pinecone_api_key"])
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

def save_to_history(question, answer):
    """Save Q&A to session state chat history"""
    st.session_state.chat_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': str(answer)
    })

def display_paper_info(paper_info):
    """Display formatted paper information"""
    st.markdown(f"""
    #### 📄 {paper_info['title']}
    
    **Authors:** {paper_info['authors']}  
    **Year:** {paper_info['year']}  
    **Journal:** {paper_info['journal']}  
    **Volume:** {paper_info['volume']}  
    **Pages:** {paper_info['pages']}  
    **DOI:** [{paper_info['doi']}](https://doi.org/{paper_info['doi']})
    
    **Keywords:** {', '.join(paper_info['keywords'])}
    
    **Abstract:**  
    {paper_info['abstract']}
    
    **Publication Details:**
    """)
    
    if 'publication_info' in paper_info:
        info = paper_info['publication_info']
        if 'issn_online' in info:
            st.markdown(f"- ISSN Online: {info['issn_online']}")
        if 'issn_print' in info:
            st.markdown(f"- ISSN Print: {info['issn_print']}")
        if 'url' in info:
            st.markdown(f"- Journal URL: [{info['url']}]({info['url']})")
        if 'published_date' in info:
            st.markdown(f"- Published: {info['published_date']}")

    if 'author_affiliation' in paper_info:
        st.markdown("**Author Affiliation:**")
        aff = paper_info['author_affiliation']
        if 'institution' in aff:
            st.markdown(f"- Institution: {aff['institution']}")
        if 'department' in aff:
            st.markdown(f"- Department: {aff['department']}")
        if 'location' in aff:
            st.markdown(f"- Location: {aff['location']}")

# Page configuration
st.set_page_config(
    page_title="Climate Change Research Assistant",
    page_icon="🌍",
    layout="wide"
)

# First, show API testing section
if not st.session_state.apis_tested:
    test_api_connections()
    if st.button("Continue to Main Application"):
        st.session_state.apis_tested = True
        st.rerun()

# Main application
if st.session_state.apis_tested:
    # Initialize query engine
    query_engine = initialize_llama_index()

    # Create tabs for main interface and paper information
    tab1, tab2 = st.tabs(["Ask Questions", "Research Papers"])

    with tab1:
        main_col, history_col = st.columns([2, 1])

        with main_col:
            st.title("Climate Change Research Assistant")
            st.markdown("""
            Ask questions about five specific scientific papers on climate change. 
            Your questions will be answered using the knowledge from these papers.
            Check the 'Research Papers' tab to see details about the source documents.
            """)

            query = st.text_input("What would you like to ask?", "")

            if st.button("Submit"):
                if not query.strip():
                    st.error("Please provide a search query.")
                else:
                    try:
                        with st.spinner('Processing your question...'):
                            response = query_engine.query(query)
                            st.markdown("### Answer:")
                            st.markdown(f">{response}")
                            save_to_history(query, response)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

        with history_col:
            st.markdown("### Chat History")
            
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.session_state.chat_history:
                for i, qa in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {qa['question'][:50]}...", expanded=(i == 0)):
                        st.markdown(f"**Time:** {qa['timestamp']}")
                        st.markdown(f"**Question:** {qa['question']}")
                        st.markdown(f"**Answer:** {qa['answer']}")
            else:
                st.info("No questions asked yet. Try asking something!")

    with tab2:
        st.title("Source Documents")
        st.markdown("""
        ### About the Research Papers
        These are the scientific papers that form the knowledge base for this assistant.
        Understanding their content will help you ask more specific questions.
        """)
        
        for paper_id, info in PAPERS_INFO.items():
            with st.expander(f"📚 {info['title']}", expanded=False):
                display_paper_info(info)
                st.markdown("**Suggested Citation:**")
                citation = f"{info['authors']}. ({info['year']}). {info['title']}. {info['journal']}, {info['volume']}, {info['pages']}."
                st.code(citation)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit | Using GPT-4 and LlamaIndex</p>
    </div>
    """, unsafe_allow_html=True)
