import os
from fastapi import FastAPI, Request
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from pydantic import BaseModel
import glob

# ⚙️ Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ULTRAMSG_TOKEN = os.getenv("ULTRAMSG_TOKEN")
ULTRAMSG_INSTANCE_ID = os.getenv("ULTRAMSG_INSTANCE_ID")

# ✅ Load PDFs from local /docs folder
def load_docs_from_folder(folder_path="docs"):
    print("📄 Loading documents from local folder...")
    documents = []
    for file_path in glob.glob(f"{folder_path}/*.pdf"):
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Failed to load {file_path}: {e}")
    if not documents:
        raise ValueError("⚠️ No documents were loaded. Make sure your PDFs are text-based.")
    return documents

# ⛓️ Setup chain
def setup_chain():
    docs = load_docs_from_folder()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    return db, chain

# 🌐 FastAPI app
app = FastAPI()
db, chain = setup_chain()

# 📩 WhatsApp message input
class Message(BaseModel):
    to: str
    message: str

# 📬 WhatsApp route
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    msg = data.get("message", "")
    sender = data.get("from", "")
    print(f"📥 Message from {sender}: {msg}")
    docs = db.similarity_search(msg)
    answer = chain.run(input_documents=docs, question=msg)
    print(f"🤖 Answer: {answer}")
    return {"reply": answer}
