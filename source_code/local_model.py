from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_PATH = "dataset/resume_model"  # your trained model path

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def generate_draft(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=300,
        do_sample=True,
        temperature=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
