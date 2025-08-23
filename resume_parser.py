import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import streamlit as st
from langchain.prompts import PromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

PROMPT_TEMPLATE = """
You are an expert resume parser. Given the resume text, extract the following fields and return a single valid JSON object:

{{
  "Name": "...",
  "Email": "...",
  "Phone": "...",
  "LinkedIn": "...",
  "Skills": [...],
  "Education": [...],
  "Experience": [...],
  "Projects": [...],
  "Certifications": [...],
  "Languages": [...]
}}

Rules:
- If a field cannot be found, set its value to "No idea".
- Return ONLY valid JSON (no extra commentary).
- Keep lists as arrays, and keep Experience/Projects as arrays of short strings.

Resume text:
{text}
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])

def resume_file_type(uploaded_file):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(temp_path)
    else:
        return None

    return loader.load()


def main():
    st.title("Resume Parser")
    uploaded_file = st.file_uploader("Upload resume", type=['txt','docx','pdf'])

    if uploaded_file:
        with st.spinner("Loading Resume..."):
            docs = resume_file_type(uploaded_file)
            if not docs:
                st.error("Unsupported file type")
                return 
        
        st.subheader("Extracted Text (Preview)")
        text_value = "\n\n".join([d.page_content for d in docs])[:4000]
        st.text_area("Preview", value=text_value, height=200)

        if st.button("Ask LLM"):
            with st.spinner("Sending to LLM..."):
                full_text = "\n\n".join([d.page_content for d in docs])
                full_prompt = PROMPT_TEMPLATE.format(text=full_text)
                response = llm.invoke(full_prompt)
                
                try:
                    parsed_json = json.loads(response.content)
                    st.json(parsed_json)
                except json.JSONDecodeError:
                    st.write(response.content)


if __name__ == "__main__":
    main()
