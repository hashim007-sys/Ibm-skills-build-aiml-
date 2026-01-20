# Ibm-skills-build-aiml-
Ai powered resume and cover letter generator and ranker


📌 Project Overview
This project is an AI-driven recruitment assistant that automates three critical hiring tasks:
Resume Ranking – Ranks resumes against a job description using Machine Learning
AI Resume Generation – Generates professional, industry-ready resumes
AI Cover Letter Generation – Creates tailored cover letters using AI text generation
The system combines Machine Learning (TF-IDF + Cosine Similarity), Natural Language Processing, and Generative AI to simulate an ATS (Applicant Tracking System) used in real-world recruitment platforms.


🎯 Key Features
📊 Machine Learning–based Resume Ranking
🧠 Skill Matching & Experience Analysis
✍️ AI-generated Resumes & Cover Letters
🖥️ Streamlit Web Interface
📄 PDF & Text Resume Support
🔌 LLM Integration (Transformer + API-based AI)


🧠 Does this project use Machine Learning?
✅ Yes
ML techniques used:
TF-IDF Vectorization – Converts text into numerical features
Cosine Similarity – Measures resume–JD relevance
Score Normalization & Weighted Ranking
This makes the system data-driven and scalable, unlike rule-based systems.


🏗️ System Architecture

Job Description (JD)
        ↓
Text Preprocessing (NLP)
        ↓
TF-IDF Vectorization (ML)
        ↓
Cosine Similarity
        ↓
Skill & Experience Extraction
        ↓
Weighted Scoring
        ↓
Resume Ranking (CSV Output)



for generation

User Details
        ↓
AI Writer (Transformer / API-based)
        ↓
Resume Generator
        ↓
Cover Letter Generator



⚙️ Installation & Setup


1️⃣ Create Virtual Environment

python3 -m venv env
source env/bin/activate

2️⃣ Install Dependencies

pip install -r requirements.txt

🚀 How to Run the Project

---first train the model

🧪 Model Training
Dataset:
Resume text → Professional resume pairs
Source: Kaggle / Custom processed dataset

Training:
python dataset/train_model.py
The trained model helps the system understand resume structure and language patterns.
trained model=dataset/resume_model
takes 15-25 mins

---second insert api key

in ai_writer.py replace GROQ_API_KEY = "YOUR GROQ API KEY HERE" with GROQ_API_KEY = "your actual groq api key"


----now the project is ready to run.

🔹 Resume Ranking (Machine Learning)
python main.py --jd jd.txt --resumes resumes

📌 Output:
artifacts/ranking.csv

🔹 Run Web Application (Resume + Cover Letter)
streamlit run source_code/app.py

📌 Output:
artifacts/generated_resume_app.txt
artifacts/generated_cover_letter_app.txt

important note- for now the output will be generated only inside vscode folder not in streamlit user interface.download option can be easily integrated later.


📌 Features:
Enter student details
Upload Job Description
Generate Resume & Cover Letter using AI

📊 Sample Ranking Output
Candidate	Similarity	SkillMatch%	FinalScore
resume1.txt	0.4001	83.3%	0.90
resume3.txt	0.0640	33.3%	0.25
resume2.txt	0.0159	16.7%	0.00


🧠 AI Resume & Cover Letter Generation
Uses Transformer-based model trained on resume datasets
Optionally enhanced using API-based LLM (GROQ / Gemini)
Generates:
Professional summaries
Bullet-point achievements
Role-specific cover letters




🔮 Future Enhancements

📄 PDF Resume Export
🧠 Fine-tuned LLM for better generation
📊 Resume Analytics Dashboard
🌐 Cloud Deployment
🧾 ATS Keyword Optimization
🤖 Multi-role Job Support


🎓 Academic Relevance

✔ Machine Learning
✔ Natural Language Processing
✔ Generative AI
✔ Real-world ATS Simulation
✔ End-to-End AI System

🏁 Conclusion
This project demonstrates a complete AI-based hiring solution that integrates:
ML for ranking
NLP for understanding
AI for content generation
It closely mirrors industry recruitment tools, making it highly suitable for final-year projects, internships, and portfolios.
