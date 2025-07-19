from fastapi import FastAPI, Request
from pdf_loader import load_pdf_data
from ultramsg_api import send_whatsapp_message
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os

app = FastAPI()

# Load FAISS index
db = FAISS.load_local("vectorstore", OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]))
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
chain = load_qa_chain(llm, chain_type="stuff")

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    message = data["data"]["body"]
    sender = data["data"]["from"]

    docs = db.similarity_search(message)
    answer = chain.run(input_documents=docs, question=message)

    send_whatsapp_message(sender, answer)
    return {"status": "sent"}