from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

def load_pdf_data(pdf_folder, api_key):
    import glob
    from langchain.text_splitter import CharacterTextSplitter

    all_docs = []
    for pdf_path in glob.glob(f"{pdf_folder}/*.pdf"):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_docs.extend(documents)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(all_docs)

    db = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=api_key))
    db.save_local("vectorstore")
