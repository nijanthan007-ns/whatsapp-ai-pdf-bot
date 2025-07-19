import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import requests
from bs4 import BeautifulSoup

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ULTRAMSG_INSTANCE_ID = os.environ.get("ULTRAMSG_INSTANCE_ID")
ULTRAMSG_TOKEN = os.environ.get("ULTRAMSG_TOKEN")

def download_google_drive_folder(folder_url, download_path="pdfs"):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    response = requests.get(folder_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.select("a")
    for link in links:
        href = link.get("href", "")
        if "/file/d/" in href:
            file_id = href.split("/file/d/")[1].split("/")[0]
            file_name = link.text.strip() or f"{file_id}.pdf"
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            file_path = os.path.join(download_path, file_name)
            with requests.get(download_url, stream=True) as r:
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
    print("✅ PDFs downloaded from Google Drive.")

def load_documents(pdf_folder="pdfs"):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            raw_docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(raw_docs)
            documents.extend(chunks)
    return documents

app = FastAPI()

class WhatsAppMsg(BaseModel):
    from_: str
    body: str

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    msg = data.get("body")
    sender = data.get("from")

    print(f"📩 Received: {msg}")

    docs = vectorstore.similarity_search(msg)
    response = chain.run(input_documents=docs, question=msg)

    reply_url = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE_ID}/messages/chat"
    payload = {
        "token": ULTRAMSG_TOKEN,
        "to": sender,
        "body": response
    }
    requests.post(reply_url, data=payload)

    return {"success": True}

print("📥 Downloading PDFs from Google Drive folder...")
download_google_drive_folder("https://drive.google.com/drive/folders/1dx5CkR3no8J1wjWR4ZVcK6HTbSqlm09R")

print("📄 Loading documents...")
documents = load_documents()

if not documents:
    raise ValueError("⚠️ No documents were loaded. Make sure the PDFs are text-based and public.")

print(f"✅ Loaded {len(documents)} chunks.")

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embedding)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
