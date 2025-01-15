# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import time
from datetime import datetime

def log_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Change to TinyBERT
model_name = "distilbert-base-uncased"  # Only 4.4M parameters vs DistilBERT's 67M
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=5,
    problem_type="single_label_classification"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# %%
import json
import os

from datasets import load_dataset, DatasetDict

# Reduce dataset size
num_train = 10000  # Reduced from 100
num_val = 5000    # Reduced from 20
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 'raw_review_Industrial_and_Scientific')
dataset['full'] = dataset['full'].select(range(num_train + num_val))

# %%
import random

random.seed(0)


def set_labels(ex):
    return {**ex, "label": int(ex["rating"] - 1), "text": ex["text"]}


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def process_ds(ds):
    ds = ds.map(set_labels)
    ds = ds.map(tokenize, batched=True, batch_size=None)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds.shuffle(seed=42)

# First, process the dataset
dataset = dataset.map(set_labels)
dataset = dataset.map(tokenize, batched=True, batch_size=None)

# Format the dataset with correct types
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
    output_all_columns=False
)

# Convert to train and validation sets
train_ds = DatasetDict({'train': dataset['full'].select(range(num_train))})
val_ds = DatasetDict({'validation': dataset['full'].select(range(num_train, num_train + num_val))})

#%%

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    ddp_find_unused_parameters=False,
    logging_first_step=True,
    disable_tqdm=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds['train'],
    eval_dataset=val_ds['validation'],
    compute_metrics=compute_metrics,
)

result = trainer.train()



# %%
# Save the trained model
trainer.save_model("star_classifier_output")
#%%
# Load the saved model
loaded_model = AutoModelForSequenceClassification.from_pretrained("star_classifier_output")
loaded_model.eval()  # Set to evaluation mode

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = loaded_model.to(device)

# %%
