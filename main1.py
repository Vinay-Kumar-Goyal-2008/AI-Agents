import json
import os
import hashlib
import requests
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf(url):
    os.makedirs("pdf_cache", exist_ok=True)
    name = hashlib.md5(url.encode()).hexdigest() + ".pdf"
    path = os.path.join("pdf_cache", name)

    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)

    loader = PyMuPDFLoader(path)
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

with open("papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_chunks = []

for i, item in enumerate(data):
    docs = load_pdf(item["pdf_url"])
    chunks = split_docs(docs)
    all_chunks.extend(chunks)
    print(i, "done")
print('Embedding the data')
print(len(all_chunks))
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 64
    }
)
print('total no. of chunks',len(all_chunks))
print('Storing documents')
vector_store = FAISS.from_documents(all_chunks, embedder)

vector_store.save_local("faiss_index")