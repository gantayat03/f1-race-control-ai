import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv("env/.env")

def run_ingestion():
    data_dir = "data"
    all_documents = []

    print(f"Scanning {data_dir} for rulebooks...")
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            print(f"Reading {filename}...")
            loader = PyPDFLoader(os.path.join(data_dir, filename))
            all_documents.extend(loader.load())

    print(f"Total pages loaded: {len(all_documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} searchable chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Updating Vector Database with Technical and Sporting rules...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
 
    print("Success! Both Technical and Sporting regulations are now indexed.")

if __name__ == "__main__":
    run_ingestion()