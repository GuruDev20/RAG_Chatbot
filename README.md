# RAG Chatbot from Scratch (Python + Ollama + ChromaDB)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** from scratch using Python.  
The chatbot answers user questions based on the content of **multiple PDF documents** by retrieving relevant information and generating grounded responses using a **locally hosted LLM via Ollama**.

The implementation avoids high-level RAG frameworks and focuses on understanding the **core concepts and pipeline** behind RAG systems.

---

## Tech Stack

- **Python** – Core programming language  
- **Ollama** – Local runtime for embeddings and LLM inference  
- **ChromaDB** – Vector database for semantic search  
- **PyPDF** – PDF text extraction  

---

## How to Run the Project

### 1. Add Documents

Place your PDF files inside the `Documents/` folder.

> Note: The `Documents/` folder is ignored in Git to avoid committing data files.

---

### 2. Install and Start Ollama

Ensure Ollama is installed and running on system.

Pull the required models:
```
ollama pull nomic-embed-text
ollama pull llama3
```

### 3. Install Dependencies
```
pip install pypdf chromadb ollama
```

### 4. Run the Chatbot
```
python3 main.py
```
