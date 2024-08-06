# LangChain-PDF-Assistant with Streamlit

## Description
This project allows you to upload a PDF file and ask questions about its content. The system extracts text from the PDF, splits it into manageable chunks, and uses LangChain's Retrieval-Augmented Generation (RAG) to provide answers to your questions. It utilizes OpenAI's GPT-3.5-turbo model for answering the questions and Streamlit for building the web application.

## Installation
Install the dependencies in the requirements.txt file by running the following command:
```
pip install -r requirements.txt
```

## Environment Variables
Set the environment variables in a .env file
```
OPENAI_KEY = <your-openai-api-key>
```

## Project Execution
1. To run the project, enter the following command:
```
streamlit run main.py
```
2. Go to the URL provided by Streamlit in your browswer. 
3. Using the file uploader, upload a PDF. Once the PDF is uploaded, ask the assistant your questions on the PDF and it will provide the answers based on the extracted text from the PDF.

## Functions
- `extract_text_from_pdf` function: reads the uploaded PDF file and extracts text from each page
- `split_text_into_documents` function: splits the extracted text into smaller chunks
- `create_vectorstore` function: creates a vector store from the document chunks
- `setup_prompt_and_chain` function: sets up the prompt template and the question-answering chain
- `main` function: handles the Streamlit interface, allowing users to upload a PDF, ask questions, and view the assistant's responses

