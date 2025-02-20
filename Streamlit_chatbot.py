import openai
import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
import json
from datetime import datetime

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Document information
PAPERS_INFO = {
    "paper1.pdf": {
        "title": "Climate Change Adaptation and Historic Settlements: Evidence from the Old Town of Corfu",
        "authors": "Eleni Maistrou, Vasiliki Pougkakioti, Miltiadis Lazoglou",
        "year": "2023",
        "journal": "American Journal of Climate Change",
        "volume": "12",
        "pages": "418-455",
        "doi": "10.4236/ajcc.2023.123020",
        "keywords": ["Historic Settlements", "Archaeological Site", "Vulnerability", "Climate Change", "Adaptation Strategy"],
        "abstract": "The Old Town of Corfu is an excellent example of a historic town and a World Heritage Site, distinguished by its authentic and unique character, as reflected in its Venetian-era fortifications and extensive historic building stock. Simultaneously, the Old Town of Corfu is also a vibrant modern city vulnerable to various pressures, including climate change. This paper aims to evaluate the effects of climate change on this modern city monument, assess its vulnerability using the Intergovernmental Panel on Climate Change's methodology, and develop a comprehensive set of adaptation proposals. The methodology of this paper is based on the analysis of climate data for the Old Town of Corfu, from which the assessment of the extreme weather events and climate changes that pose the greatest threat to the Old Town and the assessment of its vulnerability to these threats are derived. The dense geometrical characteristics of the city's structure, the intense pathology observed in the materials and structures of the historic building stock, problems in the existing electromechanical infrastructure, and the poor management of issues such as increased tourism and heavy traffic congestion are the primary factors that make the Old Town of Corfu vulnerable to the effects of climate change.",
        "publication_info": {
            "issn_online": "2167-9509",
            "issn_print": "2167-9495",
            "url": "https://www.scirp.org/journal/ajcc",
            "published_date": "September 27, 2023"
        }
    }
}

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
    
    **Publication Details:**
    - ISSN Online: {paper_info['publication_info']['issn_online']}
    - ISSN Print: {paper_info['publication_info']['issn_print']}
    - Journal URL: [{paper_info['publication_info']['url']}]({paper_info['publication_info']['url']})
    - Published Date: {paper_info['publication_info']['published_date']}
    
    **Keywords:** {', '.join(paper_info['keywords'])}
    
    **Abstract:**  
    {paper_info['abstract']}
    """)

# Page configuration
st.set_page_config(
    page_title="Climate Change Research Assistant",
    page_icon="🌍",
    layout="wide"
)

# Connecting OpenAI API
openai.api_key = st.secrets["api_secret"]

# Create tabs for main interface and paper information
tab1, tab2 = st.tabs(["Ask Questions", "Research Papers"])

with tab1:
    # Create two columns: main content and chat history
    main_col, history_col = st.columns([2, 1])

    with main_col:
        # Title and description
        st.title("Climate Change Research Assistant")
        st.markdown("""
        Ask questions about scientific papers on climate change. 
        Your questions will be answered using the knowledge from these papers.
        Check the 'Research Papers' tab to see details about the source documents.
        """)

        # Question input
        query = st.text_input("What would you like to ask?", "")

        # Submit button with error handling
        if st.button("Submit"):
            if not query.strip():
                st.error("Please provide a search query.")
            else:
                try:
                    # Show loading spinner
                    with st.spinner('Processing your question...'):
                        # Connecting large language model
                        Settings.llm = OpenAI(temperature=0.2, model="gpt-4-1106-preview")
                        
                        # Loading and indexing data
                        documents = SimpleDirectoryReader('./data').load_data()
                        index = VectorStoreIndex.from_documents(documents)
                        
                        # Generating answer
                        query_engine = index.as_query_engine()
                        response = query_engine.query(query)

                        # Display the response in a nice format
                        st.markdown("### Answer:")
                        st.markdown(f">{response}")
                        
                        # Save to chat history
                        save_to_history(query, response)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with history_col:
        st.markdown("### Chat History")
        
        # Add a clear history button
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        # Display chat history
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
    Each paper is presented with full bibliographic information and abstract.
    """)
    
    # Display paper information in expandable sections
    for paper_id, info in PAPERS_INFO.items():
        with st.expander(f"📚 {info['title']}", expanded=False):
            display_paper_info(info)
            
            # Add a citation format section
            st.markdown("**Suggested Citation:**")
            st.code(
                f"{info['authors']}. ({info['year']}). {info['title']}. "
                f"{info['journal']}, {info['volume']}, {info['pages']}. "
                f"https://doi.org/{info['doi']}"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | Using GPT-4 and LlamaIndex</p>
</div>
""", unsafe_allow_html=True)
