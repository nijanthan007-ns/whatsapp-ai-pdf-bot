import os
import requests
import tempfile
from fastapi import FastAPI, Request
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from bs4 import BeautifulSoup

# CONFIG
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
ULTRAMSG_INSTANCE_ID = os.getenv("ULTRAMSG_INSTANCE_ID")
ULTRAMSG_TOKEN = os.getenv("ULTRAMSG_TOKEN")
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1dx5CkR3no8J1wjWR4ZVcK6HTbSqlm09R"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None

def download_gdrive_folder(gdrive_url):
    print("Downloading PDFs from Google Drive folder...")
    response = requests.get(gdrive_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_links = ["https://drive.google.com/uc?id=" + tag['href'].split("/")[5] for tag in soup.find_all("a") if "/file/d/" in tag.get("href", "")]

    temp_dir = tempfile.mkdtemp()
    for link in pdf_links:
        file_id = link.split("id=")[-1]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        file_path = os.path.join(temp_dir, file_id + ".pdf")
        with requests.get(download_url, stream=True) as r:
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

    return temp_dir

def build_vectorstore():
    global vectorstore
    pdf_dir = download_gdrive_folder(GDRIVE_FOLDER_URL)

    all_docs = []
    for pdf_file in os.listdir(pdf_dir):
        loader = PyPDFLoader(os.path.join(pdf_dir, pdf_file))
        all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store built successfully!")

@app.on_event("startup")
async def startup_event():
    build_vectorstore()

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    global vectorstore
    data = await request.json()
    message = data.get("message", "")
    sender = data.get("sender", "")

    if not message or not sender:
        return JSONResponse(content={"error": "Invalid message format"}, status_code=400)

    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa_chain.run(message)

    response_url = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE_ID}/messages/chat"
    payload = {
        "token": ULTRAMSG_TOKEN,
        "to": sender,
        "body": answer
    }
    requests.post(response_url, data=payload)
    return {"status": "sent", "answer": answer}

@app.get("/")
def root():
    return {"status": "running"}
