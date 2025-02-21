# Climate Change Research Assistant

## Overview
The Climate Change Research Assistant is an interactive tool built using **Streamlit** that allows users to ask questions and receive responses based on five specific scientific papers related to climate change. This application uses **GPT-4** and **LlamaIndex** to query and generate responses from the research papers. Users can also explore detailed information about the papers, including titles, authors, abstracts, and publication details.

## Features
- **Ask Questions:** Users can ask questions about the content of five scientific papers on climate change.
- **Research Papers Tab:** Detailed information about each paper, including authors, abstracts, keywords, and publication details.
- **Chat History:** Users can track their questions and received answers, with an option to clear the history.
- **Interactive Interface:** Built with **Streamlit**, offering a user-friendly interface for browsing papers and interacting with the research assistant.

## Requirements
- Python 3.x
- Streamlit
- OpenAI GPT-4 API (requires API key)
- LlamaIndex
- Additional libraries: `openai`, `os`, `json`, `datetime`

## Setup Instructions
1. Clone the repository or download the code to your local environment.
2. Install the required libraries using pip:
    ```bash
    pip install openai streamlit llama_index
    ```
3. Obtain your **OpenAI API key** and add it to the **Streamlit secrets** configuration as `api_secret`. You can set up Streamlit secrets by creating a `secrets.toml` file:
    ```toml
    [general]
    api_secret = "your_openai_api_key_here"
    ```
4. Add the necessary research papers to the `./data` directory. The papers should be in PDF format.
5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## How It Works
- **Query Engine:** Users submit questions through the text input. The app loads and indexes the research papers using **LlamaIndex**, which is an efficient way to manage and query documents.
- **Response Generation:** The question is processed by the GPT-4 model to generate an appropriate response based on the indexed documents.
- **Chat History:** The system keeps track of questions and answers, storing them in the session state. Users can clear the history or expand to view previous interactions.

## Source Papers
The assistant is based on five scientific papers related to climate change. Users can explore each paper's details, including:
- **Title**
- **Authors**
- **Year**
- **Journal**
- **DOI**
- **Keywords**
- **Abstract**

These papers are stored in the `PAPERS_INFO` dictionary and provide the knowledge base for answering questions.

## Customization
- Add more papers: Extend the `PAPERS_INFO` dictionary to include additional research papers.
- Modify the LlamaIndex setup: Adjust the `SimpleDirectoryReader` and `VectorStoreIndex` settings to handle different formats or storage options for your documents.
- Tweak the GPT-4 settings: Modify the model or temperature settings in the `OpenAI` configuration to adjust the response behavior.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or contributions, feel free to reach out to the repository owner or create an issue on GitHub.

---

*Built with Streamlit, using GPT-4 and LlamaIndex.*
