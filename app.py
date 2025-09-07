import os
import tempfile
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
import faiss

from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

load_dotenv()

class MatchResult(BaseModel):
    match_percentage: int = Field(..., description="Overall match percentage between resume and JD")
    matching_skills: List[str] = Field(..., description="Skills present in both resume and JD")
    missing_skills: List[str] = Field(..., description="Skills required in JD but missing in resume")
    suggestions: List[str] = Field(..., description="Actionable improvements for candidate-job fit")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def compute_similarity(resume_text: str, jd_text: str) -> float:
    resume_emb = embedder.encode([resume_text])
    jd_emb = embedder.encode([jd_text])

    dim = resume_emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(resume_emb)

    D, I = index.search(jd_emb, k=1)
    similarity_score = 1 / (1 + D[0][0])
    return round(float(similarity_score), 3)

llm =  ChatGoogleGenerativeAI(model='gemini-2.0-flash', temprature=0.5)

prompt_template = """
You are an expert career advisor. Compare the following Resume and Job Description.
The cosine similarity score from embeddings is: {similarity_score}

Return structured JSON strictly in this format:
{{
  "match_percentage": <int between 0-100>,
  "matching_skills": [list of matching skills],
  "missing_skills": [list of missing skills],
  "suggestions": [list of suggestions]
}}

Resume:
{resume}

Job Description:
{jd}
"""

prompt = PromptTemplate(
    input_variables=["resume", "jd", "similarity_score"],
    template=prompt_template,
)

chain = llm | prompt 

st.title("📄 Resume & Job Description Matcher (MiniLM + FAISS + Gemini)")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description", height=200)

if resume_file and jd_text:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(resume_file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    resume_docs = loader.load()
    resume_text = " ".join([doc.page_content for doc in resume_docs])

    similarity_score = compute_similarity(resume_text, jd_text)

    with st.spinner("Analyzing..."):
       raw_result = chain.invoke(
    prompt.format(resume=resume_text, jd=jd_text, similarity_score=similarity_score)
)


    try:
        structured_result = MatchResult.parse_raw(raw_result)
    except Exception:
        st.error("⚠️ Could not parse LLM output. Showing raw result.")
        st.json(raw_result)
        structured_result = None

    if structured_result:
        st.subheader("🔎 Match Analysis")
        st.metric("Match Percentage", f"{structured_result.match_percentage}%")
        st.progress(structured_result.match_percentage / 100)

        st.write("✅ **Matching Skills:**", structured_result.matching_skills)
        st.write("❌ **Missing Skills:**", structured_result.missing_skills)
        st.write("📌 **Suggestions:**", structured_result.suggestions)

        st.caption(f"Embedding similarity score: {similarity_score}")
