# Virtual_JalilMahmud_careerAI

A conversational AI chatbot app designed specifically to assist in presenting Jalil Mahmud's career qualifications and how he fits the team and job. The chatbot leverages a RAG (Retrieval-Augmented Generation) approach using open-source technologies—integrating HuggingFace models and FAISS for efficient document retrieval and memory search, hosted with a visually appealing and modern HTML/CSS frontend.

---

## Features

- **Career-Focused Chatbot**: Demonstrates Jalil Mahmud's career experience and how it aligns with potential roles.
- **RAG-based Question Answering**: Combines LLM reasoning with retrieval from indexed documents for accurate and context-aware responses.
- **HuggingFace Models**: Uses `mistralai/Mistral-7B-Instruct-v0.2` for text generation and `xlm-roberta-base` for multilingual PDF embedding.
- **Multilingual Support**: Capable of processing documents and queries in multiple languages, making it suitable for diverse teams.
- **Telegram Integration**: Automatically sends user queries and chatbot responses to Telegram for real-time updates.
- **Modern and Responsive UI**: Features a visually appealing, futuristic design with improved chat visualization and message flow.
- **Deployable on Render**: Ready to be hosted with persistent database support.

---

## Live Demo

Try the chatbot here: [Virtual_JalilMahmud_careerAI](https://chatbot-panel-app.onrender.com)

---

## Tech Stack

- Python
- Flask for backend server
- HTML and CSS for UI
- HuggingFace Transformers & Embeddings
- FAISS (vector database) for efficient similarity search
- LangChain for language model integration
- PyPDFLoader for PDF text extraction
- Telegram API for message forwarding

---

## Project Status

- The chatbot UI is continually being improved for better user experience.
- Background and styling are updated to feature a dynamic, futuristic design without using images.
- Enhanced message visualization and user interaction flow.

---

Contributions and suggestions are welcome! Let's make career presentation smarter and more interactive with AI!
