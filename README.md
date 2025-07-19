# WhatsApp AI Assistant Bot (PDF Knowledge Base)

## Features
- Answers your questions based on PDF files using GPT-4.
- WhatsApp integration using Ultramsg.
- Deployable on Railway or Render.

## Setup
1. Set environment variables using `.env`.
2. Run `pdf_loader.py` once to create FAISS vector index from PDFs.
3. Deploy using `uvicorn main:app --host 0.0.0.0 --port 8000`.