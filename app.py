import os
import torch
import faiss
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# Telegram
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# Hugging Face Client
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Embedding model
print("Loading the multilingual embedding model...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")

# FAISS and metadata
index_file = "pdf_embeddings.faiss"
metadata_file = "pdf_metadata.npy"
pdf_folder = "pdfs"

# Chunking
chunk_size = 256
chunk_overlap = 50


def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send message to Telegram: {e}")


def query_huggingface(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print("Error extracting content:", e)
        return "Error generating response from LLM."


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings


def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def read_pdf_files(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            chunks = chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                documents.append((filename, i, chunk))
    return documents


def create_faiss_index_from_pdfs(pdf_folder, index_file, metadata_file):
    documents = read_pdf_files(pdf_folder)
    print(f"Found {len(documents)} chunks to embed...")

    embeddings = []
    for doc in documents:
        emb = embed_text(doc[2])
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_file)
    np.save(metadata_file, np.array(documents, dtype=object), allow_pickle=True)

    print("FAISS index and metadata created and saved.")
    return index, documents


def initialize_faiss():
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        print("Loading existing FAISS index and metadata...")
        index = faiss.read_index(index_file)
        metadata = np.load(metadata_file, allow_pickle=True).tolist()
    else:
        print("FAISS files not found. Creating new index from PDFs...")
        index, metadata = create_faiss_index_from_pdfs(pdf_folder, index_file, metadata_file)
    return index, metadata


def search_faiss(query, index, metadata, top_k=5):
    embedding = embed_text(query)
    if len(embedding.shape) == 1:
        embedding = np.expand_dims(embedding, axis=0)
    distances, indices = index.search(embedding, top_k)
    retrieved_chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            retrieved_chunks.append(metadata[idx])
    return retrieved_chunks


def generate_response(query, retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, (list, tuple)) and len(chunk) > 2 and isinstance(chunk[2], str):
            if chunk[2].strip():
                context_parts.append(chunk[2])
    
    context = "\n".join(context_parts)

    if not context.strip():
        return "Sorry, I couldn’t find relevant context to answer your question."

    prompt = (
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "You are representer of Jalil Mahmud. If the context does not contain relevant information, just say you don't know. "
        "Do not mention file names or sources. Keep the answer clear and to the point."
    )

    return query_huggingface(prompt)


@app.route('/')
def index():
    return render_template("app.html")


@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get("query")
    send_to_telegram(f"User Input: {user_query}")

    retrieved_chunks = search_faiss(user_query, index, metadata)
    print("Retrieved chunks:", retrieved_chunks)  # debug print
    response = generate_response(user_query, retrieved_chunks)

    send_to_telegram(f"LLM Response: {response}")
    return jsonify({"response": response})


# Load or build FAISS index
index, metadata = initialize_faiss()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

