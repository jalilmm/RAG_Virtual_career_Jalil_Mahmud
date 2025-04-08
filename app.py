from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from transformers import AutoModel, AutoTokenizer
import os
import faiss
import glob
import numpy as np
import torch
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Telegram credentials from environment variables
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# Initialize the InferenceClient with the API token
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# FAISS index and metadata file paths
index_file = "pdf_embeddings.faiss"
metadata_file = "pdf_metadata.npy"

# Chunking settings
chunk_size = 256
chunk_overlap = 50

# Load the multilingual embedding model once
print("Loading the multilingual embedding model...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModel.from_pretrained("xlm-roberta-base")


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
        model="mistralai/Mistral-7B-Instruct-v0.2",
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


@app.route('/')
def index():
    return render_template("app.html")


@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get("query")
    send_to_telegram(f"User Input: {user_query}")

    retrieved_chunks = search_faiss(user_query, index, metadata)
    response = generate_response(user_query, retrieved_chunks)

    send_to_telegram(f"LLM Response: {response}")
    return jsonify({"response": response})


def initialize_faiss():
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        print("Loading existing FAISS index and metadata...")
        index = faiss.read_index(index_file)
        metadata = np.load(metadata_file, allow_pickle=True).tolist()
    else:
        index = faiss.IndexFlatL2(768)
        metadata = []
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
    # Combine the chunk texts without explicitly mentioning the file or chunk number
    context = "\n".join([chunk[2] for chunk in retrieved_chunks])
    prompt = (
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        
        "If the context does not contain relevant information, just say you don't know. "
        "Do not mention file names or sources. Keep the answer clear and to the point."
    )
    
    return query_huggingface(prompt)



index, metadata = initialize_faiss()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
