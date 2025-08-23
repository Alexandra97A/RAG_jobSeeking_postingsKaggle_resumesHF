import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import torch

class JobMatchingRAG:
    def __init__(self, 
                 embed_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model="mistralai/Mistral-7B-Instruct-v0.2",
                 jobs_file="./Data/kaggle_marketing_sample_for_trulia_com-real_estate__20190901_20191031__30k_data.csv",
                 resumes_file="./Data/HF_master_resumes.jsonl"):
        
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.jobs_file = jobs_file
        self.resumes_file = resumes_file
        
        # Initialize components
        self.embedder = None
        self.local_llm = None
        self.index = None
        self.jobs = None
        self.resumes = None
        
    def load_data(self):
        """Load and preprocess job and resume data"""
        print("Loading job data...")
        self.jobs = pd.read_csv(self.jobs_file)
        print(f"Initial jobs loaded: {len(self.jobs)}")
        self.jobs = self.jobs.dropna(subset=["Job Description"]).reset_index(drop=True)
        
        print("Loading resume data...")
        self.resumes = pd.read_json(self.resumes_file, lines=True)
        
        # Process text
        self.jobs["body"] = self.jobs["Job Title"] + "\n" + self.jobs["Job Description"]
        self.resumes["body"] = self.resumes.apply(self._extract_resume_text, axis=1)
        
        print(f"Processed: {len(self.jobs)} jobs, {len(self.resumes)} resumes")
        
    def _extract_resume_text(self, row):
        """Extract meaningful text from resume data structure"""
        text_parts = []
        
        # Extract skills
        try:
            if row['skills'] is not None and isinstance(row['skills'], dict):
                skills_text = []
                for category, skills in row['skills'].items():
                    if isinstance(skills, dict):
                        for subcategory, skill_list in skills.items():
                            if isinstance(skill_list, list):
                                for skill in skill_list:
                                    if isinstance(skill, dict) and 'name' in skill:
                                        skills_text.append(skill['name'])
                                    elif isinstance(skill, str):
                                        skills_text.append(skill)
                if skills_text:
                    text_parts.append("Skills: " + ", ".join(skills_text))
        except (KeyError, TypeError):
            pass
        
        # Extract experience
        try:
            if row['experience'] is not None and isinstance(row['experience'], list):
                exp_text = []
                for exp in row['experience']:
                    if isinstance(exp, dict):
                        if 'company' in exp and exp['company'] != 'Unknown':
                            exp_text.append(f"Worked at {exp['company']}")
                        if 'role' in exp and exp['role'] != 'Unknown':
                            exp_text.append(f"Role: {exp['role']}")
                if exp_text:
                    text_parts.append("Experience: " + ". ".join(exp_text))
        except (KeyError, TypeError):
            pass
        
        # Extract education
        try:
            if row['education'] is not None and isinstance(row['education'], list):
                edu_text = []
                for edu in row['education']:
                    if isinstance(edu, dict) and 'degree' in edu:
                        degree = edu['degree']
                        if isinstance(degree, dict):
                            if 'level' in degree and degree['level'] != 'Unknown':
                                edu_text.append(degree['level'])
                            if 'field' in degree and degree['field'] != 'Unknown':
                                edu_text.append(degree['field'])
                if edu_text:
                    text_parts.append("Education: " + " in ".join(edu_text))
        except (KeyError, TypeError):
            pass
        
        return "\n".join(text_parts) if text_parts else "No information available"
    
    def setup_embedding_model(self):
        """Initialize the embedding model"""
        print(f"Loading embedding model: {self.embed_model}")
        self.embedder = SentenceTransformer(self.embed_model)
        
    def setup_llm(self):
        """Initialize the language model"""
        print(f"Loading LLM: {self.llm_model}")
        self.local_llm = pipeline(
            "text-generation", 
            model=self.llm_model,
            device=0 if torch.cuda.is_available() else -1,
            max_length=384,
            do_sample=True,
            temperature=0.5,
        )
    
    def build_job_index(self):
        """Create FAISS index for job embeddings"""
        if self.embedder is None:
            raise ValueError("Embedding model not initialized. Call setup_embedding_model() first.")
        
        print("Generating job embeddings...")
        jobs_embeddings = self.embedder.encode(
            self.jobs["body"].tolist(), 
            show_progress_bar=True, 
            batch_size=32
        )
        
        print("Building FAISS index...")
        jobs_embeddings = np.array(jobs_embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(jobs_embeddings.shape[1])
        self.index.add(jobs_embeddings)
        
    def match_jobs_for_resume(self, resume_text, k=5):
        """Find top-k matching jobs for a given resume"""
        if self.index is None:
            raise ValueError("Job index not built. Call build_job_index() first.")
        
        # Embed resume
        resume_vec = self.embedder.encode([resume_text]).astype('float32')
        
        # Search
        D, I = self.index.search(resume_vec, k)
        
        return self.jobs.iloc[I[0]]
    
    def explain_match(self, resume_text, job_text):
        """Generate AI explanation for job-resume match"""
        if self.local_llm is None:
            raise ValueError("LLM not initialized. Call setup_llm() first.")
        
        prompt = (
            f"Resume:\n{resume_text}\n\n"
            f"Job Description:\n{job_text}\n\n"
            "Summarize skill matches, missing elements, and whether this candidate is a good fit."
        )
        
        result = self.local_llm(prompt, num_return_sequences=1)
        return result[0]["generated_text"].strip()
    
    def initialize_system(self):
        """Initialize the complete RAG system"""
        self.load_data()
        self.setup_embedding_model()
        self.setup_llm()
        self.build_job_index()
        print("RAG system initialized successfully!")
    
    def get_resume_by_index(self, idx):
        """Get resume text by index"""
        if self.resumes is None:
            raise ValueError("Resume data not loaded.")
        return self.resumes.iloc[idx]['body']
    
    def get_resume_count(self):
        """Get total number of resumes"""
        return len(self.resumes) if self.resumes is not None else 0