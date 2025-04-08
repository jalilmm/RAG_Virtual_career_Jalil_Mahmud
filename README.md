# Virtual_JalilMahmud_careerAI

A conversational AI chatbot app designed specifically to assist in presenting Jalil Mahmud's career qualifications and how he fits the team and job. The chatbot leverages a RAG (Retrieval-Augmented Generation) approach using open-source technologies—integrating HuggingFace models and FAISS for efficient document retrieval and memory search, hosted with a visually appealing and modern HTML/CSS frontend.

With AWS S3 integration, the application retrieves the necessary FAISS files and metadata directly from the cloud, ensuring seamless access across deployments.

---

## Features

- **Career-Focused Chatbot**: Demonstrates Jalil Mahmud's career experience and how it aligns with potential roles.
- **RAG-based Question Answering**: Combines LLM reasoning with retrieval from indexed documents for accurate and context-aware responses.
- **HuggingFace Models**: Uses `mistralai/Mistral-7B-Instruct-v0.2` for text generation and `xlm-roberta-base` for multilingual PDF embedding.
- **Multilingual Support**: Capable of processing documents and queries in multiple languages, making it suitable for diverse teams.
- **Telegram Integration**: Automatically sends user queries and chatbot responses to Telegram for real-time updates (to see model behavior).
- **AWS S3 Integration**: FAISS files and metadata are securely stored and retrieved from AWS S3, ensuring efficient and scalable data access.
- **Modern and Responsive UI**: Features a visually appealing, futuristic design with improved chat visualization and message flow.
- **Deployable on Render**: Ready to be hosted with persistent database support.

---

## Live Demo

Try the chatbot here: [Virtual_JalilMahmud_careerAI](https://chatbot-panel-app.onrender.com)

---

## Tech Stack

- **Python**: Core language for building the application.
- **Flask**: Backend server for routing and processing requests.
- **HTML and CSS**: For building the modern UI that the user interacts with.
- **HuggingFace Transformers & Embeddings**: For natural language understanding and question-answering.
- **FAISS**: Used as a vector database for efficient document similarity search.
- **AWS S3**: Cloud storage for FAISS index and metadata files, ensuring scalable data access.
- **LangChain**: For language model integration and handling the chain of tasks.
- **PyPDFLoader**: For extracting text from PDF documents.
- **Telegram API**: For sending real-time updates of user queries and chatbot responses to Telegram.

---

## AWS Integration

- The application retrieves the necessary FAISS files and metadata from **AWS S3** if they are not already available locally. 
- **S3 Bucket**: The files are uploaded to an S3 bucket (`mycareerllm`), making the data accessible across different environments without pushing data to GitHub.
- **IAM Role & Policies**: Secure access to the S3 bucket is handled via IAM roles, ensuring that only authorized users can interact with the data.

---

## Project Status

- The chatbot UI is continually being improved for a better user experience.
- Background and styling have been updated to feature a dynamic, futuristic design without using images.
- Enhanced message visualization and user interaction flow.
- AWS S3 integration allows for easy access to large datasets (FAISS index and metadata) from anywhere.

---

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/jalilmm/RAG_Virtual_career_Jalil_Mahmud.git
   cd RAG_Virtual_career_Jalil_Mahmud
