# 📘 UG Rulebook Chatbot (Gradio + Colab)

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built entirely in **Google Colab**, powered by **Meta’s LLaMA 2 7B Chat** model, using a PDF (UG Rulebook) as the knowledge base. The chatbot answers queries from users based only on the contents of the PDF.

---

## 🚀 Features

- 📄 PDF parsing using `PyMuPDF`
- 🧠 Text splitting + embeddings with `LangChain`
- 🔍 Vector similarity search using `FAISS`
- 🤖 Response generation using `meta-llama/Llama-2-7b-chat-hf`
- 💬 Local UI using `Gradio` for interactive Q&A

---

## 🧰 Dependencies

Install all required libraries in the Colab notebook:

```python
!pip install transformers accelerate bitsandbytes huggingface_hub
!pip install PyMuPDF langchain faiss-cpu sentence-transformers gradio
```

## Structure
- LLM model initialization
- Pdf loading and parsing
- Text splitter
- Vector database
- Searching and passing the context to the prompt
- Custom prompt passing to the model and processing the output
- Displaying output using UI
- Evaluate how the LLM is doing and identify quantifiable metrics for that


## 💬 Example Query
- Q: What is the minimum CPI required for branch change?

- ✅ A: Returns accurate answer only from the rulebook context, such as:

- "The minimum CPI required for branch change is 8.0..."


## 📌 Notes
- The chatbot only answers questions based on the PDF content.

- LLaMA 2 model weights are downloaded on the fly and require ~10GB of space.

- Built and tested entirely in Google Colab — no local installation needed.

## 🙌 Acknowledgments
- Hugging Face for LLaMA and transformers

- LangChain for text chunking and FAISS support

- Gradio for simple UI deployment

---
