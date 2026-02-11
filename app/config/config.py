import os
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
HUGGINGFACE_REPO_ID="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
DB_CHROMA_PATH="vectorstore/chroma_db"
DATA_PATH="data/"
CHUNK_SIZE=500
CHUNK_OVERLAP=50
