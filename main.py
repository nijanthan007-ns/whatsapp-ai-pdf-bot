from fastapi import FastAPI, Request
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

app = FastAPI()

@app.post("/webhook")
async def webhook(req: Request):
    return {"status": "ok"}

db = FAISS.load_local(
    "vectorstore",
    OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
    allow_dangerous_deserialization=True
)
