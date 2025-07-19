from fastapi import FastAPI, Request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

app = FastAPI()

# Load the FAISS vector store with deserialization allowed
db = FAISS.load_local(
    "vectorstore",
    OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
    allow_dangerous_deserialization=True
)

@app.get("/")
def root():
    return {"message": "Bot is running!"}

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    print("Incoming request:", data)

    # Example response logic — customize for Ultramsg or 360Dialog
    return {"status": "received"}
