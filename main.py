import os
import zipfile
import gdown
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Step 1: Download and extract PDFs from Google Drive ZIP
def download_and_extract_zip():
    file_id = "1_igDhNY5GstbpAMxvmM1B4gTtFsXwrv_"  # Your GDrive file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = "manuals.zip"
    extract_path = "pdfs"

    if not os.path.exists(zip_path):
        print("📥 Downloading ZIP from Google Drive...")
        gdown.download(url, zip_path, quiet=False)

    if not os.path.exists(extract_path):
        print("📦 Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

# Step 2: Load PDFs and build vectorstore
def build_vectorstore():
    download_and_extract_zip()
    data_path = "pdfs"
    all_docs = []

    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_path, filename)
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

# Step 3: Setup FastAPI app
app = FastAPI()
vectorstore = build_vectorstore()
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = load_qa_chain(llm, chain_type="stuff")

# Step 4: WhatsApp / POST endpoint
@app.post("/ask")
async def ask_question(request: Request):
    try:
        body = await request.json()
        question = body.get("question")

        if not question:
            return JSONResponse(status_code=400, content={"error": "Missing 'question' field"})

        docs = vectorstore.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)
        return {"answer": answer}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
