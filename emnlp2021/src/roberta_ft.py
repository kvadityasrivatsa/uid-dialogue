import sys
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import os

if len(sys.argv) != 2:
  print("Usage: python roberta_ft.py <dataset_path>")
  sys.exit(1)

dataset_path = sys.argv[1]
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

# Load and prepare the dataset
df = pd.read_csv(dataset_path)
sentences = df['sentence'].tolist()

dataset = Dataset.from_dict({"sentence": sentences})
train_test_split = dataset.train_test_split(test_size=0.3, seed=42)

dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

# Load RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=["sentence"])

# Training arguments
output_dir = f"./roberta-{dataset_name}-finetuned"
final_model_dir = f"./roberta-{dataset_name}-finetuned_final"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
)

# Data collator with mlm=True for masked language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(final_model_dir)

# Evaluate and print the perplexity
eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
