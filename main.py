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

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

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

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = get_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

def generate_answer(query, context_chunks):
    context_text = "\n\n".join(context_chunks)

    prompt = f"""
        You are an information extraction system.
        Rules:
        - Answer ONLY using the provided context
        - Be concise and specific
        - Do NOT add explanations
        - If the answer is not explicitly stated, say "Not found in the documents"

        Context:
        {context_text}

        Question:
        {query}

        Answer (1-3 sentences max):
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]

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

    print("\nRAG Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break

        relevant_chunks = retrieve_relevant_chunks(user_query)
        answer = generate_answer(user_query, relevant_chunks)

        print("\nBot:")
        print(answer)
        print("-" * 50)
