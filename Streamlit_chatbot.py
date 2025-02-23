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

# Papers Information Dictionary
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
    },
    "paper2.pdf": {
        "title": "The Analysis of Global Warming Patterns from 1970s to 2010s",
        "authors": "Ali Cheshmehzangi",
        "year": "2020",
        "journal": "Atmospheric and Climate Sciences",
        "volume": "10",
        "pages": "392-404",
        "doi": "10.4236/acs.2020.103022",
        "keywords": ["Global Warming", "Climate Change", "Global Warming Patterns", "Atmospheric Temperature", "Ocean Surface Temperature", "Global Warming Impacts"],
        "abstract": "While global warming is only one part of climate change effects, it poses the highest risk to our habitats and ecologies. It is alarming that global warming has heightened in multiple locations and is intensified since the early 1970s. Since then, there are certain global warming patterns that could guide us with an overview of what mitigation and adaptation strategies should be developed in the future decades. There are certain regions affected more than another, and there are certain patterns with adverse effects on regions, sub-regions, and even continents. This study provides an insightful analysis of recent global warming patterns, those that are affecting us the most with regional climate change of different types, upsurge in frequency and intensity of natural disasters, and drastic impacts on our ecosystems around the world. By analysing the global warming patterns of these last four decades, this research study sheds light on where these patterns are coming from, how they are developing, and what are their impacts. This study is conducted through grey literature and analysis of the recorded global warming data publicly available by the NASA-GISS data centre for global temperature. This brief—but comprehensive—analysis helps us to have a better understanding of what comes next for global warming impacts, and how we should ultimately react. The study contributes to the field by discovering three key points analysed based on available data and literature on recorded global temperature, including: differences between north and south hemispheres, specific patterns due to ocean surface temperature increase, and recent impacts on particular regions. The study concludes with the importance of global scale analysis to have a more realistic understanding of the global warming patterns and their impacts on all living habitats.",
        "publication_info": {
            "issn_online": "2160-0422",
            "issn_print": "2160-0414",
            "url": "https://www.scirp.org/journal/acs",
            "published_date": "July 1, 2020"
        },
        "author_affiliation": {
            "institution": "The University of Nottingham Ningbo China",
            "department": "Department of Architecture and Built Environment",
            "location": "Ningbo, China"
        }
    },
    "paper3.pdf": {
        "title": "Warming Power of CO₂ and H₂O: Correlations with Temperature Changes",
        "authors": "Paulo Cesar Soares",
        "year": "2010",
        "journal": "International Journal of Geosciences",
        "volume": "1",
        "pages": "102-112",
        "doi": "10.4236/ijg.2010.13014",
        "keywords": ["Global Warming", "CO₂", "Vapor Greenhouse"],
        "abstract": "The dramatic and threatening environmental changes announced for the next decades are the result of models whose main drive factor of climatic changes is the increasing carbon dioxide in the atmosphere. Although taken as a premise, the hypothesis does not have verifiable consistence. The comparison of temperature changes and CO₂ changes in the atmosphere is made for a large diversity of conditions, with the same data used to model climate changes. Correlation of historical series of data is the main approach. CO₂ changes are closely related to temperature. Warmer seasons or triennial phases are followed by an atmosphere that is rich in CO₂, reflecting the gas solving or exsolving from water, and not photosynthesis activity. Interannual correlations between the variables are good. A weak dominance of temperature changes precedence, relative to CO₂ changes, indicate that the main effect is the CO₂ increase in the atmosphere due to temperature rising. Decreasing temperature is not followed by CO₂ decrease, which indicates a different route for the CO₂ capture by the oceans, not by gas re-absorption. Monthly changes have no correspondence as would be expected if the warming was an important absorption-radiation effect of the CO₂ increase. The anthropogenic wasting of fossil fuel CO₂ to the atmosphere shows no relation with the temperature changes even in an annual basis. The absence of immediate relation between CO₂ and temperature is evidence that rising its mix ratio in the atmosphere will not imply more absorption and time residence of energy over the Earth surface. This is explained because band absorption is nearly all done with historic CO₂ values. Unlike CO₂, water vapor in the atmosphere is rising in tune with temperature changes, even in a monthly scale. The rising energy absorption of vapor is reducing the outcoming long wave radiation window and amplifying warming regionally and in a different way around the globe.",
        "publication_info": {
            "url": "http://www.SciRP.org/journal/ijg",
            "published_date": "November 2010"
        },
        "author_affiliation": {
            "institution": "Federal University of Parana (UFPR)",
            "department": "Earth Sciences",
            "location": "Curitiba, Brazil"
        }
    },
    "paper4.pdf": {
        "title": "The Impact of Energy Produced by Civilization on Global Warming",
        "authors": "Vladimir Kh. Dobruskin",
        "year": "2022",
        "journal": "Open Journal of Ecology",
        "volume": "12",
        "pages": "325-332",
        "doi": "10.4236/oje.2022.126019",
        "keywords": ["Thermodynamics", "Global Warming", "Energy of Civilization", "Climate"],
        "abstract": "The thermodynamic approach to the evolution of human society shows that the energy generated by civilization disrupts the thermal balance of the Earth. This energy did not exist before the advent of civilization; it practically does not affect the thermal radiation of the planet and dissipates in the atmosphere in the form of heat, increasing the kinetic energy of gas molecules and, consequently, their temperature. Since air molecules cannot leave the Earth due to gravity, excess heat accumulates on the planet and contributes to global warming. A quantitative assessment of the effect is given. An analogy can be made: the energy generated by humanity heats the atmosphere, as a furnace heats a dwelling.",
        "publication_info": {
            "issn_online": "2162-1993",
            "issn_print": "2162-1985",
            "url": "https://www.scirp.org/journal/oje",
            "published_date": "June 20, 2022"
        },
        "author_affiliation": {
            "institution": "Independent Researcher",
            "location": "Beer Yacov, Israel"
        }
    },
    "paper5.pdf": {
        "title": "The World's Largest Lakes Water Level Changes in the Context of Global Warming",
        "authors": "Valery S. Vuglinsky, Maria R. Kuznetsova",
        "year": "2019",
        "journal": "Natural Resources",
        "volume": "10",
        "pages": "29-46",
        "doi": "10.4236/nr.2019.102003",
        "keywords": ["Large Lakes", "Water Level", "Changes", "Global Warming"],
        "abstract": "The article is focused on the assessment of changes in the average annual water levels of large lakes of the planet in the changing climate conditions characteristic of the recent decades. Eight large lakes, i.e. Baikal, Balkhash, Superior, Issyk-Kul, Ladoga, Onega, Ontario, and Erie, located on the territory of Eurasia and North America, were chosen as the research objects. They were selected because of the availability of a long-term observations series of the water level. As is known, long-term changes in the lake's water level result from variation in the water volume. The latter depends on the ratios between the water balance components of the lake that have developed during a given year, which, in turn, reflect the climatic conditions of the respective years. The features of the water balance structure of the above-mentioned lakes and the intra-annual course of the water level are considered. The available long-term records of observational data on all selected lakes and their stations were divided into two periods: from 1960 to 1979 (the period of stationary climatic situation) and from 1980 to 2008 (the period of non-stationary climatic situation). The homogeneity and significance of trends in the long-term water level series of records have been estimated. It has been established that over the second period the nature and magnitude of the lakes water levels variations differ significantly. For lakes Balkhash, Issyk-Kul, Ladoga, Superior, and Erie, there is a general tendency for a decrease in water levels. For the remaining three lakes (Baikal, Onega, and Ontario), the opposite tendency has been noted: the levels of these lakes increased. Quantitatively, the range of changes in water levels on the lakes in question over the period of 1980-2008 ranged from −4 cm to +26 cm.",
        "publication_info": {
            "issn_online": "2158-7086",
            "issn_print": "2158-706X",
            "url": "http://www.scirp.org/journal/nr",
            "published_date": "February 26, 2019"
        },
        "author_affiliation": {
            "institution": "St. Petersburg State University",
            "location": "St. Petersburg, Russia"
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
    
    **Keywords:** {', '.join(paper_info['keywords'])}
    
    **Abstract:**  
    {paper_info['abstract']}
    
    **Publication Details:**
    """)
    
    # Display publication info if available
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

    # Display author affiliation if available
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
        Ask questions about five specific scientific papers on climate change. 
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
            st.rerun()
        
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
    Understanding their content will help you ask more specific questions.
    """)
    
    # Display paper information in expandable sections
    for paper_id, info in PAPERS_INFO.items():
        with st.expander(f"📚 {info['title']}", expanded=False):
            display_paper_info(info)
            # Add citation format
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
