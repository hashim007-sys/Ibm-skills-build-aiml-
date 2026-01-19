from groq import Groq

# 🔑 Paste your Groq API key INSIDE the quotes
GROQ_API_KEY = "YOUR GROQ API KEY HERE"

client = Groq(api_key=GROQ_API_KEY)

def generate_text(prompt: str) -> str:
    """
    Generates professional-quality text using Groq LLaMA model.
    Used for resume and cover letter generation.
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a professional resume and cover letter writer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=700
    )

    return response.choices[0].message.content.strip()
