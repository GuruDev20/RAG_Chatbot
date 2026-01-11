import os
from pypdf import PdfReader
import chromadb
import ollama

chromadb_client = chromadb.Client()
collection = chromadb_client.get_or_create_collection(name="document_chunks")

def read_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text

def load_documents(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading document: {filename}")
            all_text += read_pdf(file_path) + "\n"
    return all_text

def split_text(text,chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def get_embeddings(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

def store_chunks_in_chromadb(chunks):
    for i, chunk in enumerate(chunks):
        embedding = get_embeddings(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )

if __name__ == "__main__":
    folder_path = "Documents"

    print("Loading documents from folder:", folder_path)
    documents_text = load_documents(folder_path)

    print("Splitting text into chunks...")
    text_chunks = split_text(documents_text)
    print(f"Total chunks created: {len(text_chunks)}")

    print("Storing chunks in ChromaDB...")
    store_chunks_in_chromadb(text_chunks)
    print("All chunks stored successfully.")