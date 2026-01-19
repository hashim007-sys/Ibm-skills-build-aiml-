import streamlit as st
import json
import os
from resume_generator import generate_resume
from cover_letter_generator import generate_cover_letter

st.title("AI Resume & Cover Letter Builder")

# --- Student Information Input ---
st.header("Student Information")
name = st.text_input("Name", "Hashim")
education = st.text_input("Education", "B.Tech CSE")
skills = st.text_area("Skills (comma separated)", "Python, Machine Learning, NLP")
projects = st.text_area("Projects (comma separated)", "AI Resume Ranker, Rubiks Cube Solver")
experience = st.text_input("Experience", "Internship")

student_data = {
    "name": name,
    "education": education,
    "skills": [s.strip() for s in skills.split(",")],
    "projects": [p.strip() for p in projects.split(",")],
    "experience": experience
}

# Save student data to temporary JSON
os.makedirs("data", exist_ok=True)
student_file = "data/student_temp.json"
with open(student_file, "w", encoding="utf-8") as f:
    json.dump(student_data, f, indent=2)

# --- Resume & Cover Letter Generation ---
st.header("Generate Resume & Cover Letter")
jd_file = st.file_uploader("Upload Job Description (txt)", type="txt")

if st.button("Generate Resume & Cover Letter") and jd_file:
    # Save uploaded JD
    jd_path = os.path.join("data", "jd_uploaded.txt")
    with open(jd_path, "wb") as f:
        f.write(jd_file.getbuffer())

    # Ensure artifacts folder exists
    os.makedirs("artifacts", exist_ok=True)

    # Output paths
    resume_path = "artifacts/generated_resume_app.txt"
    cover_path = "artifacts/generated_cover_letter_app.txt"

    # Generate files
    generate_resume(student_file, resume_path)
    generate_cover_letter(student_file, jd_path, cover_path)

    st.success("Files Generated!")
    st.write(f"- Resume: {resume_path}")
    st.write(f"- Cover Letter: {cover_path}")
