#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
HF_TOKEN = os.getenv("HF_TOKEN")

# Then use HF_TOKEN instead of hardcoding the token
login(HF_TOKEN)

# Set up CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
torch.cuda.empty_cache()
#%%# Load model directly
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=HF_TOKEN)

# Move model to device
model = model.to(device)
#%% Function to generate text
def generate_text(prompt, max_length=100):
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

#%% Example usage
prompt = "Write a short story about a robot"
generated_output = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_output}")
# %%
# Load Amazon Reviews dataset
from datasets import load_dataset

# Load the dataset (this might take a while as it's a large dataset)
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 'raw_review_Industrial_and_Scientific') # Load first 1000 examples

# Print basic information about the dataset
print("\nDataset structure:")
print(dataset)
#%%
# Display first few examples from the dataset
for i, example in enumerate(dataset['full'].select(range(3))):
    print(f"\n--- Item {i+1} ---")
    for key, value in example.items():
        # Truncate long values for better readability
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        elif isinstance(value, list) and len(str(value)) > 100:
            value = str(value[:2]) + "..."
        print(f"{key}: {value}")


# %%
from datasets import DatasetDict, Dataset

dataset['full'] = dataset['full'].select(range(100))
dataset
