from langchain_community.document_loaders import PyPDFLoader
import logging

logging.basicConfig(level=logging.INFO)

file_path = "../PIC18F45880_user_manual.compressed.pdf"
loader = PyPDFLoader(file_path)

pdf_pages_list = loader.load()

logging.info(f"Number of pages in pdf: {len(pdf_pages_list)}")
