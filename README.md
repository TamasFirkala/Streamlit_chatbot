# Climate Change Research Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climate-research-assistant.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered research assistant that helps you extract insights from climate change scientific papers. Ask questions about climate research in natural language and get accurate answers based on a curated collection of peer-reviewed papers.

## 🌍 Overview

This application provides a chat interface to interact with scientific papers on climate change, allowing users to:

- Ask questions about climate change research in natural language
- Get answers based on peer-reviewed scientific papers
- Browse detailed information about the source documents
- View and manage chat history

The assistant leverages a vector database backend to provide accurate, citation-based responses without hallucinating information.

## 🔧 Features

- **Question-answering interface**: Ask natural language questions about climate change research
- **Paper information browser**: View detailed metadata about the source documents
- **Semantic search**: Find relevant information across multiple papers with vector search
- **Chat history**: Keep track of previous queries and answers
- **Citation visibility**: Understand where information is coming from

## 📚 Knowledge Base

The assistant is built on a curated collection of climate change research papers:

1. "Climate Change Adaptation and Historic Settlements: Evidence from the Old Town of Corfu" (2023)
2. "The Analysis of Global Warming Patterns from 1970s to 2010s" (2020)
3. "Warming Power of CO₂ and H₂O: Correlations with Temperature Changes" (2010)
4. "The Impact of Energy Produced by Civilization on Global Warming" (2022)
5. "The World's Largest Lakes Water Level Changes in the Context of Global Warming" (2019)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-3.5 Turbo
- **Embedding Model**: OpenAI text-embedding-3-small
- **Vector Database**: Pinecone
- **RAG Framework**: LlamaIndex

## 🚀 Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and environment

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/yourusername/climate-research-assistant.git
cd climate-research-assistant
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys
```
openai_api_key=your_openai_api_key
pinecone_api_key=your_pinecone_api_key
pinecone_environment=your_pinecone_environment
pinecone_index_name=your_pinecone_index_name
```

4. Run the Streamlit app
```bash
streamlit run app.py
```

## 📋 Usage

### Asking Questions

1. Navigate to the "Ask Questions" tab
2. Type your question about climate change research
3. Click "Submit" to get an answer based on the research papers
4. View previous questions and answers in the chat history panel

### Exploring Papers

1. Navigate to the "Research Papers" tab
2. Browse the available papers and expand entries to see details
3. Review abstracts, keywords, author information, and citation details

## 🧠 How It Works

1. **Vector Embedding**: The scientific papers are processed and transformed into vector embeddings using OpenAI's embedding model
2. **Vector Database**: These embeddings are stored in Pinecone for efficient retrieval
3. **Semantic Search**: When a question is asked, LlamaIndex performs semantic search to find relevant sections of the papers
4. **Context-Aware Response**: GPT-3.5 Turbo generates a response based on the retrieved context
5. **History Management**: All interactions are stored in the session state for reference

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The scientific papers used in this project are publicly available and properly cited
- Built using Streamlit, OpenAI, Pinecone, and LlamaIndex

---

Developed with ❤️ for climate research accessibility
