import streamlit as st
from rag_jobs import JobMatchingRAG

# Initialize RAG system with caching
@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    rag = JobMatchingRAG()
    rag.initialize_system()
    return rag

def main():
    st.title("Resume-to-Job Matching: Local RAG Demo")
    
    # Load RAG system
    with st.spinner("Loading RAG system..."):
        rag = load_rag_system()
    
    st.success("RAG system loaded successfully!")
    
    # Resume selection from dataset OR file upload
    option = st.radio("Choose resume source:", ["From dataset", "Upload resume"])
    
    sample_resume_body = None
    
    if option == "From dataset":
        max_idx = rag.get_resume_count() - 1
        idx = st.number_input(
            f"Select Resume Index [0, {max_idx}]:", 
            min_value=0, 
            max_value=max_idx, 
            value=0
        )
        sample_resume_body = rag.get_resume_by_index(idx)
        
    elif option == "Upload resume":
        upload = st.file_uploader("Upload resume (plain text)", type=["txt"])
        if upload:
            sample_resume_body = upload.read().decode('utf-8')
    
    if sample_resume_body:
        st.subheader("Resume Preview")
        st.code(sample_resume_body)
        st.write("---")
        
        # Job matching controls
        k = st.slider("Number of job matches", min_value=1, max_value=10, value=5)
        
        if st.button("Find Matching Jobs"):
            with st.spinner("Finding job matches..."):
                matches = rag.match_jobs_for_resume(sample_resume_body, k=k)
            
            st.subheader("Job Matches")
            
            for i, (_, job) in enumerate(matches.iterrows()):
                st.markdown(f"### Match #{i+1}: {job['Job Title']}")
                st.markdown(job['Job Description'][:700] + "...")
                
                with st.spinner(f"Generating explanation for match #{i+1}..."):
                    explanation = rag.explain_match(sample_resume_body, job["Job Description"])
                
                st.info(explanation)
                st.write("---")

if __name__ == "__main__":
    main()