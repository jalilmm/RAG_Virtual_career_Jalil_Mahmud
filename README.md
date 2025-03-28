# Virtual_JalilMahmud_careerAI

A conversational AI chatbot app designed for interactive career-related queries. This project uses a RAG (Retrieval-Augmented Generation) approach with a fully open-source stack—leveraging HuggingFace models and FAISS for smart document and memory search, all hosted with a modern Panel UI.

---

## Features

- **RAG-based Question Answering**: Combines LLM reasoning with retrieval from indexed documents.
- **Persistent Memory**: Saves chat history and embeds conversation data with vector similarity search.
- **HuggingFace Models**: Uses `mistralai/Mistral-7B-Instruct-v0.2` for generation and `sentence-transformers/all-mpnet-base-v2` for embeddings.
- **Chat History Similarity**: Loads most similar past user queries from chat logs to improve continuity.
- **Telegram Integration**: Sends each user query and bot response directly to your Telegram.
- **simple UI with Panel**: Styled chat boxes, loading spinner, and responsive layout.(working on improvement)
- **Deployable on Render**: Fully web-deployable with persistent DB support.

---

## 🛠 Tech Stack

- Python
- [Panel](https://panel.holoviz.org/)
- HuggingFace Transformers & Embeddings
- FAISS (vector database)
- LangChain
- PyPDFLoader
- Telegram API (optional)
- ![image](https://github.com/user-attachments/assets/096c87c5-0f09-4971-b533-ba6eddf00e89)

