import os
import shutil
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Constants
PDF_FOLDER = "docs"

# Load documents from GitHub docs folder
def load_documents():
    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError("📁 'docs' folder not found. Please ensure your PDFs are in a 'docs' directory.")

    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, filename))
            docs = loader.load()
            documents.extend(docs)
    return documents

# Split documents into chunks
def split_documents(documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Create vector store
def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# Create QA chain
def create_chain():
    llm = ChatOpenAI(temperature=0.2)
    prompt_template = """Use the following context to answer the question in simple terms:

{context}

Question: {question}
Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

# Load knowledge base
print("📄 Loading documents from GitHub 'docs' folder...")
docs = load_documents()
if not docs:
    raise ValueError("⚠️ No documents were loaded. Make sure the 'docs' folder contains valid PDF files.")
chunks = split_documents(docs)
db = create_vectorstore(chunks)
chain = create_chain()
print("✅ Knowledge base ready.")

# WhatsApp handler route
@app.post("/")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    user_msg = data.get("message")
    if not user_msg:
        return JSONResponse(content={"reply": "No message received."})

    docs = db.similarity_search(user_msg)
    response = chain.run(input_documents=docs, question=user_msg)
    return JSONResponse(content={"reply": response})
