import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text, extract_text_from_pdf, detect_skills, extract_experience_years

def normalize_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    arr = s.astype(float).to_numpy()
    lo, hi = arr.min(), arr.max()
    if abs(hi - lo) < 1e-9:
        return pd.Series(np.ones_like(arr))
    return pd.Series((arr - lo) / (hi - lo))

def rank_resumes(jd_text: str, resume_paths: list, skills_vocab: set,
                 weights=(0.6, 0.3, 0.1), min_df=1, max_features=100000):

    docs = [jd_text]
    names = ["__JD__"]
    raw_resume_texts = []

    for p in resume_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(p)
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        raw = clean_text(text)
        docs.append(raw)
        names.append(os.path.basename(p))
        raw_resume_texts.append(raw)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features
    )

    X = vectorizer.fit_transform(docs)
    jd_vec = X[0:1]
    res_vecs = X[1:]
    sims = cosine_similarity(res_vecs, jd_vec).ravel()

    jd_skills = set(detect_skills(jd_text, skills_vocab))
    if not jd_skills:
        jd_skills = set(list(skills_vocab)[:10])

    skill_counts = []
    matched_skill_lists = []
    for txt in raw_resume_texts:
        matched = detect_skills(txt, jd_skills)
        skill_counts.append(len(matched))
        matched_skill_lists.append(matched)

    exp_years = [extract_experience_years(txt) for txt in raw_resume_texts]

    df = pd.DataFrame({
        "Candidate": names[1:],
        "Similarity": sims,
        "SkillsMatched": skill_counts,
        "SkillsRequired": [len(jd_skills)] * len(skill_counts),
        "MatchedSkillsList": [", ".join(lst) for lst in matched_skill_lists],
        "ExpYears": exp_years
    })

    sim_norm = normalize_series(df["Similarity"])
    skill_pct = (df["SkillsMatched"] / df["SkillsRequired"].replace(0, np.nan)).fillna(0)
    skill_norm = normalize_series(skill_pct)
    exp_norm = normalize_series(df["ExpYears"])

    w_sim, w_skill, w_exp = weights
    df["FinalScore"] = (w_sim * sim_norm + w_skill * skill_norm + w_exp * exp_norm).round(4)
    df["SkillMatch%"] = (skill_pct * 100).round(1)

    df = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    df["Similarity"] = df["Similarity"].round(4)
    df["ExpYears"] = df["ExpYears"].round(1)

    return df, vectorizer, jd_skills
