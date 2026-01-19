import re
from pdfminer.high_level import extract_text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()

def extract_text_from_pdf(path: str) -> str:
    try:
        return extract_text(path)
    except Exception:
        return ""

def detect_skills(text: str, skills_vocab: set) -> list:
    found = []
    text = text.lower()
    for skill in skills_vocab:
        if skill in text:
            found.append(skill)
    return sorted(list(set(found)))

def extract_experience_years(text: str) -> float:
    text = text.lower()

    # pattern like "2 years", "3+ years"
    matches = re.findall(r'(\d+)\s*\+?\s*years?', text)
    if matches:
        return float(max(map(int, matches)))

    # pattern like "2019 - 2023"
    year_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2})', text)
    if year_ranges:
        years = [int(end) - int(start) for start, end in year_ranges]
        return float(max(years))

    return 0.0
