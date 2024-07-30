import PyPDF2
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
import logging, os
import streamlit as st
import fire
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
llm_model = ChatOpenAI(model="gpt-3.5-turbo")


def main():
    st.title("PDF QnA")
    uploaded_file = st.file_uploader("Upload a pdf", type="pdf")

    if uploaded_file is not None:
        question = st.text_input("Ask a question about the pdf")

        if st.button("Submit"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            logging.info(f"Number of pages in pdf: {len(pdf_reader.pages)}")
            pdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_text(pdf_text)
            documents = [Document(page_content=split) for split in splits]
            vectorstore = Chroma.from_documents(
                documents=documents, embedding=OpenAIEmbeddings()
            )

            retriever = vectorstore.as_retriever()

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            results = rag_chain.invoke({"input": question})

            logging.info(results["context"][0].metadata)
            logging.info(results["answer"])
            st.write("Answer: ", results["answer"])


if __name__ == "__main__":
    main()
