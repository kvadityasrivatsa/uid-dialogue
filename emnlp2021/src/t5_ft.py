import sys
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import torch

if len(sys.argv) != 2:
    print("Usage: python t5_ft.py <dataset_path>")
    sys.exit(1)

dataset_path = sys.argv[1]
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

df = pd.read_csv(dataset_path)
df = df[["Source Utterance Index", "Target Utterance Index", "Utterance 1", "Utterance 2"]]

df["source"] = df["Utterance 1"]
df["target"] = df["Utterance 2"]

dataset = Dataset.from_pandas(df[["source", "target"]])
train_test_split = dataset.train_test_split(test_size=0.3, seed=42)

print(f"Train size: {len(train_test_split['train'])}")

dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "test": train_test_split["test"]
})

model_name = "t5-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = tokenizer(examples["source"], max_length=256, truncation=True, padding="max_length")
    targets = tokenizer(examples["target"], max_length=256, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, remove_columns=["source", "target"])

output_dir = f"./t5-{dataset_name}-finetuned"
final_model_dir = f"./t5-{dataset_name}-finetuned_final"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
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
    save_strategy="steps",
    load_best_model_at_end=True,
    predict_with_generate=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model(final_model_dir)

eval_results = trainer.evaluate()
print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
