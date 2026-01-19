import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
import torch

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "t5-small"
DATA_PATH = "clean_training_data.csv"
OUTPUT_DIR = "./resume_model"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 5e-5

# -----------------------------
# LOAD DATA
# -----------------------------
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# -----------------------------
# TOKENIZER & MODEL
# -----------------------------
print("🔤 Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# -----------------------------
# TOKENIZATION FUNCTION
# -----------------------------
def preprocess(batch):
    inputs = tokenizer(
        batch["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH
    )

    targets = tokenizer(
        batch["target_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_TARGET_LENGTH
    )

    inputs["labels"] = targets["input_ids"]
    return inputs

# -----------------------------
# APPLY TOKENIZATION
# -----------------------------
print("🧹 Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    fp16=False,
    report_to="none"
)

# -----------------------------
# TRAINER
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -----------------------------
# TRAIN
# -----------------------------
print("🚀 Training started...")
trainer.train()

# -----------------------------
# SAVE MODEL
# -----------------------------
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ TRAINING COMPLETE")
print(f"📁 Model saved to: {OUTPUT_DIR}")
