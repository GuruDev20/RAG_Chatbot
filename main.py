import os
from pypdf import PdfReader

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

if __name__ == "__main__":
    folder_path = "Documents"

    print("Loading documents from folder:", folder_path)
    documents_text = load_documents(folder_path)

    print("Combined Document Text:")
    print(documents_text[:1000])