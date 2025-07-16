# -*- coding: utf-8 -*-
"""Chatbot with LLaMA 2 and RAG using PDF, LangChain, FAISS and Gradio UI"""

# Required: pip install transformers accelerate bitsandbytes huggingface_hub PyMuPDF gradio faiss-cpu sentence-transformers langchain

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr

# ------------------- LOGIN -------------------
from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()

# Get token from environment
login(token=os.getenv("HF_TOKEN"))

# ------------------- MODEL SETUP -------------------
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer and config
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Initialize empty model to offload later
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Create device map (use CPU and offload to disk)
device_map = infer_auto_device_map(
    model,
    max_memory={"cpu": "16GiB"},
    no_split_module_classes=["LlamaDecoderLayer"]  # Specific to LLaMA
)

# Load model with offload
model = load_checkpoint_and_dispatch(
    model,
    model_name,
    device_map=device_map,
    offload_folder="offload"
)

# ------------------- CHATBOT PIPELINE -------------------
from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# ------------------- LOAD PDF -------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

pdf_path = "ugrulebook.pdf"
document_text = extract_text_from_pdf(pdf_path)

# ------------------- TEXT SPLITTING -------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_text(document_text)

# ------------------- VECTOR DB -------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = FAISS.from_texts(chunks, embedding_model)
vector_db.save_local("faiss_index_ug_rulebook")

# ------------------- RAG PROMPTING -------------------
def build_prompt(context, question):
    return f"""
Answer the following question using **only** the information in the context below.

Context:
{context}

Question: {question}
Answer:"""

def get_answer_from_pdf(query, k=3):
    docs = vector_db.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = build_prompt(context, query)
    response = chatbot(prompt)[0]["generated_text"]
    if "Answer:" in response:
        return response.split("Answer:")[1].strip()
    else:
        return response.strip()

# ------------------- GRADIO UI -------------------
def gradio_chatbot(query):
    return get_answer_from_pdf(query)

if __name__ == "__main__":
    interface = gr.Interface(
        fn=gradio_chatbot,
        inputs=gr.Textbox(
            lines=2,
            placeholder="Ask something from the UG Rulebook"
        ),
        outputs="text",
        title="UG Rulebook Chatbot",
        description="Ask questions based on the UG Rulebook PDF"
    )
    interface.launch(share=True)
