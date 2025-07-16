import streamlit as st
from chatbot import get_answer_from_pdf

# Page config
st.set_page_config(page_title="UG Rulebook Chatbot", layout="centered")

# Title
st.title("📘 UG Rulebook Chatbot")

st.markdown("""
Ask questions based only on the IITB UG Rulebook PDF.

**Examples:**
- How many credits for a minor?
- Can dual degree students register for a minor?
""")

# Input
query = st.text_input("Ask a question")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            answer = get_answer_from_pdf(query)
            st.success(answer)
    else:
        st.warning("Please enter a question.")
