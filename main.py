from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

app = FastAPI()

# Load environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Load PDFs from /docs
def load_pdfs():
    docs = []
    for filename in os.listdir("docs"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("docs", filename))
            docs.extend(loader.load())
    return docs

# Process & embed PDF data
print("Loading and processing PDFs...")
documents = load_pdfs()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Vector store setup
print("Creating vector store...")
vectorstore = FAISS.from_documents(chun_
