import json
import os
from local_model import generate_draft
from ai_writer import generate_text

def generate_resume(student_file, out_file):
    with open(student_file, "r", encoding="utf-8") as f:
        s = json.load(f)

    # Step 1: Draft from YOUR transformer
    draft_prompt = f"""
Generate resume draft content.
Name: {s['name']}
Education: {s['education']}
Skills: {', '.join(s['skills'])}
Projects: {', '.join(s['projects'])}
Experience: {s['experience']}
"""
    draft_text = generate_draft(draft_prompt)

    # Step 2: Professional rewrite by Groq
    final_prompt = f"""
You are a professional resume writer.

Rewrite and expand the following draft into a polished, industry-ready resume.
Use bullet points, strong action verbs, and professional tone.

Draft:
{draft_text}
"""
    final_resume = generate_text(final_prompt)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(final_resume)

    return out_file
