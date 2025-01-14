# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import time
from datetime import datetime

def log_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

log_time("Script started")

# Change to TinyBERT
model_name = "prajjwal1/bert-tiny"  # Only 4.4M parameters vs DistilBERT's 67M
log_time("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
log_time("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=5,
    problem_type="single_label_classification"
)

log_time(f"Model loaded: {model_name}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_time(f"Moving model to {device}")
model = model.to(device)
log_time("Model moved to device")

# %%
import json
import os

from datasets import load_dataset, DatasetDict

# Reduce dataset size
num_train = 20  # Reduced from 100
num_val = 5    # Reduced from 20
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 'raw_review_Industrial_and_Scientific')
dataset['full'] = dataset['full'].select(range(num_train + num_val))
#%%
dataset['full'][0]
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

# Ensure labels are long tensors
def convert_to_long(examples):
    examples["label"] = torch.tensor(examples["label"], dtype=torch.long)
    return examples

train_ds['train'] = train_ds['train'].map(convert_to_long)
val_ds['validation'] = val_ds['validation'].map(convert_to_long)

print("Dataset processed")
print(f"First training example shape: {next(iter(train_ds['train'])).keys()}")
print(f"First training example label: {next(iter(train_ds['train']))['label']}")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=0,
    weight_decay=0,
    logging_dir="./logs",
    logging_steps=1,
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=False,
    report_to="none",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    local_rank=-1,
    ddp_find_unused_parameters=False,
    logging_first_step=True,
    disable_tqdm=False,
    max_grad_norm=1.0,
)

# Add a custom training step function to debug what's happening
class DebugTrainer(Trainer):
    def training_step(self, model, inputs):
        log_time("Starting training step")
        log_time(f"Input keys: {inputs.keys()}")
        log_time(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
        
        # Get device from model's parameters instead of model.device
        device = next(model.parameters()).device
        log_time(f"Using device: {device}")
        
        # Move inputs to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        model.zero_grad()  # Ensure gradients are clean
        
        try:
            # Explicitly use CrossEntropyLoss
            criterion = torch.nn.CrossEntropyLoss()
            
            outputs = model(**inputs)
            log_time("Forward pass completed")
            
            # Calculate loss manually to ensure it's correct
            logits = outputs.logits
            labels = inputs['labels']
            loss = criterion(logits, labels)
            
            log_time(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            log_time(f"Raw loss value: {loss.item()}")
            
            # Loss should always be positive for cross entropy
            if loss.item() < 0:
                log_time("WARNING: Negative loss detected - this should never happen!")
                loss = torch.abs(loss)  # Force positive
                
            log_time(f"Final loss value: {loss.item()}")
            return loss
            
        except Exception as e:
            log_time(f"Error in training step: {str(e)}")
            return torch.tensor(0.0, device=device)

# Use our debug trainer
trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds['train'],
    eval_dataset=val_ds['validation'],
)

# Before training starts
log_time("About to start training...")
try:
    start_time = time.time()
    result = trainer.train()
    end_time = time.time()
    log_time(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Training result: {result}")
except Exception as e:
    log_time(f"Error during training: {str(e)}")
    raise e


# %%
