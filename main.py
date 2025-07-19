from fastapi import FastAPI, Request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import glob
import gdown
import requests

app = FastAPI()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDINGS = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
VECTORSTORE_PATH = "vectorstore"
DOCS_DIR = "docs"
GOOGLE_DRIVE_FOLDER_ID = "1dx5CkR3no8J1wjWR4ZVcK6HTbSqlm09R"
ULTRAMSG_INSTANCE_ID = os.environ.get("ULTRAMSG_INSTANCE_ID")
ULTRAMSG_TOKEN = os.environ.get("ULTRAMSG_TOKEN")

def download_pdfs_from_drive():
    print("📥 Downloading PDFs from Google Drive folder...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    gdown.download_folder(id=GOOGLE_DRIVE_FOLDER_ID, output=DOCS_DIR, quiet=False, use_cookies=False)
    print("✅ Download complete.")

def build_vectorstore():
    if not os.path.exists(DOCS_DIR) or len(glob.glob(f"{DOCS_DIR}/*.pdf")) == 0:
        download_pdfs_from_drive()

    all_docs = []
    pdf_files = glob.glob(f"{DOCS_DIR}/*.pdf")
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        all_docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(all_docs)

    db = FAISS.from_documents(texts, EMBEDDINGS)
    db.save_local(VECTORSTORE_PATH)
    print("✅ Vectorstore created and saved.")
    return db

# Load or create vectorstore
if os.path.exists(f"{VECTORSTORE_PATH}/index.faiss"):
    db = FAISS.load_local(
        VECTORSTORE_PATH,
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )
    print("📂 Loaded existing vectorstore.")
else:
    db = build_vectorstore()

@app.get("/")
def root():
    return {"status": "Bot is running ✅"}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("📩 Webhook received:", data)

    try:
        sender = data['data']['from']
        message = data['data']['body']

        docs = db.similarity_search(message, k=3)
        answer = docs[0].page_content if docs else "Sorry, I couldn't find anything."

        send_url = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE_ID}/messages/chat"
        payload = {
            "token": ULTRAMSG_TOKEN,
            "to": sender,
            "body": answer
        }
        r = requests.post(send_url, data=payload)
        print("📤 Sent reply:", r.text)

    except Exception as e:
        print("❌ Error:", e)

    return {"status": "ok"}