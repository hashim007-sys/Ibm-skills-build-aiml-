import json
import os
from local_model import generate_draft
from ai_writer import generate_text

def generate_cover_letter(student_file, jd_file, out_file):
    with open(student_file, "r", encoding="utf-8") as f:
        s = json.load(f)

    with open(jd_file, "r", encoding="utf-8") as f:
        jd = f.read()

    # Step 1: Draft from YOUR transformer
    draft_prompt = f"""
Generate a cover letter draft.
Candidate: {s['name']}
Skills: {', '.join(s['skills'])}
Experience: {s['experience']}
Job Description: {jd}
"""
    draft_text = generate_draft(draft_prompt)

    # Step 2: Polished rewrite by Groq
    final_prompt = f"""
Rewrite the following draft into a professional, concise cover letter.
Align it with the job description and use confident formal language.

Draft:
{draft_text}
"""
    final_letter = generate_text(final_prompt)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_letter)

    return out_file
