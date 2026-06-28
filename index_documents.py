import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

DATA_DIR = "data"
INDEX_DIR = "index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = "all-mpnet-base-v2" 

def tokenize(text):
    """Токенизация текста для BM25 (нижний регистр, токены)"""
    return [word.lower() for word in word_tokenize(text) if word.isalnum()]

def load_documents_from_dir(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith((".txt", ".md")):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Загружен текст: {filename}")
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Загружен PDF: {filename}")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Всего чанков: {len(chunks)}")
    return chunks

def main():
    documents = load_documents_from_dir(DATA_DIR)
    if not documents:
        print("Нет документов в папке data/")
        return

    chunks = split_documents(documents)
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_metadatas = [chunk.metadata for chunk in chunks]

    print("Генерация dense-эмбеддингов...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunk_texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    print("Построение BM25 индекса...")
    tokenized_chunks = [tokenize(text) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    os.makedirs(INDEX_DIR, exist_ok=True)

    # FAISS индекс
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.bin"))

    with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump({
            "chunks": chunk_texts,
            "meta": chunk_metadatas,
            "tokenized_chunks": tokenized_chunks, 
            "bm25": bm25                          
        }, f)


    print(f"   Индекс сохранён в папку '{INDEX_DIR}'")
    print(f"   Чанков: {len(chunk_texts)}")
    print(f"   Размерность эмбеддингов: {dim}")

if __name__ == "__main__":
    main()