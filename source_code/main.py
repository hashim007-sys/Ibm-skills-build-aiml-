import argparse
import glob
import os
import json
import joblib
from ranker import rank_resumes

def load_skills(path):
    if not path:
        return {
            "python","java","c++","sql","javascript",
            "machine learning","data science","nlp",
            "pandas","numpy","scikit-learn","tensorflow",
            "pytorch","flask","streamlit","docker","aws"
        }

    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

def main():
    parser = argparse.ArgumentParser("AI Resume Ranker")
    parser.add_argument("--jd", required=True, help="Path to job description")
    parser.add_argument("--resumes", required=True, help="Folder with resumes")
    parser.add_argument("--skills", default=None, help="Optional skills file")
    parser.add_argument("--out", default="artifacts/ranking.csv")
    parser.add_argument("--weights", type=float, nargs=3, default=(0.6, 0.3, 0.1))

    args = parser.parse_args()

    with open(args.jd, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # ✅ Only include .txt or .pdf files (skip directories)
    resume_paths = [
    f for f in glob.glob(os.path.join(args.resumes, "*.txt"))
    if os.path.isfile(f)
    and os.path.basename(f).lower().startswith("resume")
]

    skills = load_skills(args.skills)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df, vectorizer, jd_skills = rank_resumes(
        jd_text,
        resume_paths,
        skills,
        weights=tuple(args.weights)
    )

    df.to_csv(args.out, index=False)
    joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.joblib")

    with open("artifacts/jd_skills.json", "w", encoding="utf-8") as f:
        json.dump(sorted(jd_skills), f, indent=2)

    print("Ranking saved to", args.out)

if __name__ == "__main__":
    main()
