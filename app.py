import sys
sys.path.append('../..')

import panel as pn
import param
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import os
import json
import requests
from glob import glob
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.schema import HumanMessage, AIMessage

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MAX_TELEGRAM_MESSAGE_LENGTH = 4000

HISTORY_FILE = "chat_history.json"
CHAT_FAISS_PATH = "chat_memory_faiss"
FAISS_DB_PATH = "faiss_index"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Persistent history load/save

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

chat_history_log = load_history()

# Save chat to FAISS

def save_chat_to_faiss(history):
    if not history:
        return
    docs = [f"User: {x['user']}\nBot: {x['bot']}" for x in history]
    metadata = [{"user": x["user"], "bot": x["bot"]} for x in history]
    texts = [x["user"] for x in history]  # Embed only user questions
    vectordb = FAISS.from_texts(texts, embedding=embedding, metadatas=metadata)
    vectordb.save_local(CHAT_FAISS_PATH)

def load_chat_faiss():
    if os.path.exists(os.path.join(CHAT_FAISS_PATH, "index.faiss")):
        return FAISS.load_local(FAISS_DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)
    return None

save_chat_to_faiss(chat_history_log)

# Telegram sender

def send_telegram_message(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        chunks = [message[i:i + MAX_TELEGRAM_MESSAGE_LENGTH] for i in range(0, len(message), MAX_TELEGRAM_MESSAGE_LENGTH)]
        for chunk in chunks:
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
            response = requests.post(url, data=data)
            if response.status_code != 200:
                print(f"🚫 Failed to send chunk: {response.status_code}")
                print(response.text)

# Load documents and embed

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_and_embed(folder_path):
    if os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss")):
        print("🔁 Loading FAISS index from disk...")
        return FAISS.load_local(FAISS_DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)


    print("🧠 Creating FAISS index from documents...")
    all_chunks = []
    for file in glob(os.path.join(folder_path, "*.pdf")):
        loader = PyPDFLoader(file)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        all_chunks.extend(splits)

    vectordb = FAISS.from_documents(all_chunks, embedding)
    vectordb.save_local(FAISS_DB_PATH)
    return vectordb

# Prompt Template
retriever_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know.
Keep the answer concise. Avoid follow-up questions. Always say 'thanks for asking!' at the end.
{context}
Question: {question}
Helpful Answer:
"""
)

# HuggingFace LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question"
)

class cbfs(param.Parameterized):
    answer = param.String("")

    def __init__(self, **params):
        super(cbfs, self).__init__(**params)
        self.panels = []
        self.folder_path = "C:/Users/jalil/projects/NLP_playground/RAG_cv"
        self.db = load_and_embed(self.folder_path)
        self.chat_db = load_chat_faiss()
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": retriever_prompt_template},
        )

    def convchain(self, query):
        if not query:
            return

        memory.chat_memory.messages = []
        if self.chat_db:
            docs = self.chat_db.similarity_search(query, k=3)
            for doc in docs:
                meta = doc.metadata
                memory.chat_memory.add_user_message(meta["user"])
                memory.chat_memory.add_ai_message(meta["bot"])

        result = self.qa.invoke({"question": query})
        full_answer = result.get("answer", "").strip()
        clean_answer = full_answer.split("Helpful Answer:")[-1].strip()
        self.answer = clean_answer

        chat_history_log.append({"user": query, "bot": clean_answer})
        save_history(chat_history_log)
        save_chat_to_faiss(chat_history_log)

        telegram_msg = f"📥 User prompt:\n{query}\n\n📤 Bot answer:\n{self.answer}"
        send_telegram_message(telegram_msg)

        self.panels.append(pn.Row('User:', pn.pane.Markdown(query, width=600)))
        self.panels.append(pn.Row('Answer:', pn.pane.Markdown(self.answer, width=600)))

    def clr_history(self, count=0):
        self.panels = []
        chat_box.objects = []
        memory.clear()
        chat_history_log.clear()
        save_history(chat_history_log)
        return

cb = cbfs()

submit_button = pn.widgets.Button(name="Send", button_type="primary")
inp = pn.widgets.TextInput(placeholder='Enter text here…')
chat_box = pn.WidgetBox(*cb.panels, scroll=True)

def handle_send(event):
    cb.convchain(inp.value)
    chat_box.objects = cb.panels
    inp.value = ''

submit_button.on_click(handle_send)

app = pn.Column(
    pn.Row(pn.pane.Markdown("# Virtual_JalilMahmud_careerAI")),
    pn.Row(inp, submit_button),
    pn.layout.Divider(),
    chat_box,
    pn.layout.Divider(),
)

app.servable()
