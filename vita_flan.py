import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import os

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = r"E:\project_Vita\chatbot\vitamin_NLP_1.csv"
OUTPUT_DIR = r"E:\project_Vita\flan-t5-vitamin-model"
MODEL_NAME = "google/flan-t5-base"

# ----------------------------
# LOAD DATASET
# ----------------------------
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH, quotechar='"', engine='python', on_bad_lines='skip', encoding='latin1')
print("✅ Loaded CSV successfully:", df.shape)

# Standardize columns
df.columns = [c.strip().lower() for c in df.columns]
if not all(col in df.columns for col in ["deficiency", "question", "answer"]):
    raise ValueError("CSV must contain columns: deficiency, question, answer")

df = df.dropna(subset=["deficiency", "question", "answer"]).reset_index(drop=True)
print(f"✅ Cleaned dataset: {len(df)} rows")

# Combine columns into input/output text
df["input_text"] = "Deficiency: " + df["deficiency"] + ". Question: " + df["question"]
df["target_text"] = df["answer"]

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# ----------------------------
# LOAD TOKENIZER & MODEL
# ----------------------------
print("\n🔁 Loading FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

MAX_LEN = 128

# ----------------------------
# PREPROCESS FUNCTION
# ----------------------------
def preprocess_function(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=MAX_LEN, truncation=True, padding="max_length")
    labels = tokenizer(batch["target_text"], max_length=MAX_LEN, truncation=True, padding="max_length")
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("⚙️ Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Split into train and eval
split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ----------------------------
# TRAINING SETUP
# ----------------------------
training_args = TrainingArguments(
    output_dir="./vitamin_t5_finetuned",
    save_strategy="epoch",
    eval_steps=500,   # ✅ Alternative to evaluation_strategy
    save_total_limit=2,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    push_to_hub=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ----------------------------
# TRAINING
# ----------------------------
print("\n🚀 Starting fine-tuning...")
trainer.train()

# ----------------------------
# SAVE MODEL
# ----------------------------
print("\n💾 Saving fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuned model saved at: {OUTPUT_DIR}")
