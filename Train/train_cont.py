import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
import argparse

# Parse command-line arguments for hyperparameters and dataset paths
parser = argparse.ArgumentParser(description='Fine-tune BERT model with contrastive learning.')
parser.add_argument('--train_data', type=str, required=True, help='Path to the training dataset (JSONL file).')
parser.add_argument('--test_data', type=str, required=True, help='Path to the testing dataset (JSONL file).')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs.')
parser.add_argument('--temperature', type=float, default=0.05, help='Temperature parameter for contrastive loss.')
parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping.')
parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenization.')
parser.add_argument('--model_name', type=str, default='dbmdz/bert-base-italian-xxl-uncased', help='Pre-trained model name.')
args = parser.parse_args()

# Initialize wandb
wandb.init(project='contriever-finetunev2', config=vars(args))
config = wandb.config

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModel.from_pretrained(config.model_name).to(device)

# Load datasets
train_dataset = load_dataset('json', data_files=config.train_data)['train']
test_dataset = load_dataset('json', data_files=config.test_data)['train']


def tokenize_function(examples):
    # Ensure that 'examples' is a dict of lists
    queries = examples['query']
    positives = examples['positive']
    negatives = examples['hard_negative']

    # Initialize empty lists for tokenized outputs
    anchor_input_ids = []
    anchor_attention_mask = []
    positive_input_ids = []
    positive_attention_mask = []
    negative_input_ids = []
    negative_attention_mask = []

    # Process each example in the batch
    for i in range(len(queries)):
        query = queries[i]
        positive = positives[i]
        negative = negatives[i]

        # Check if any of the texts are missing or not strings
        if not isinstance(query, str) or not isinstance(positive, str) or not isinstance(negative, str):
            continue  # Skip this example

        # Tokenize
        anchor = tokenizer(query, padding='max_length', truncation=True, max_length=config.max_length)
        pos = tokenizer(positive, padding='max_length', truncation=True, max_length=config.max_length)
        neg = tokenizer(negative, padding='max_length', truncation=True, max_length=config.max_length)

        anchor_input_ids.append(anchor['input_ids'])
        anchor_attention_mask.append(anchor['attention_mask'])
        positive_input_ids.append(pos['input_ids'])
        positive_attention_mask.append(pos['attention_mask'])
        negative_input_ids.append(neg['input_ids'])
        negative_attention_mask.append(neg['attention_mask'])

    # Return tokenized inputs
    return {
        'anchor_input_ids': anchor_input_ids,
        'anchor_attention_mask': anchor_attention_mask,
        'positive_input_ids': positive_input_ids,
        'positive_attention_mask': positive_attention_mask,
        'negative_input_ids': negative_input_ids,
        'negative_attention_mask': negative_attention_mask,
    }
# After loading the datasets
train_dataset = load_dataset('json', data_files=config.train_data)['train']
test_dataset = load_dataset('json', data_files=config.test_data)['train']

# Filter out invalid examples
def filter_valid_examples(example):
    return (
        isinstance(example['query'], str) and example['query'].strip() != '' and
        isinstance(example['positive'], str) and example['positive'].strip() != '' and
        isinstance(example['hard_negative'], str) and example['hard_negative'].strip() != ''
    )

train_dataset = train_dataset.filter(filter_valid_examples)
test_dataset = test_dataset.filter(filter_valid_examples)

# Proceed with tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['query', 'positive', 'hard_negative'])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['query', 'positive', 'hard_negative'])


# Set format for PyTorch
columns = [
    'anchor_input_ids', 'anchor_attention_mask',
    'positive_input_ids', 'positive_attention_mask',
    'negative_input_ids', 'negative_attention_mask'
]
train_dataset.set_format(type='torch', columns=columns)
test_dataset.set_format(type='torch', columns=columns)

# DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=config.learning_rate)
total_steps = len(train_dataloader) * config.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# Contrastive loss function
criterion = nn.CrossEntropyLoss()

def compute_embeddings(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embeddings

# Early stopping parameters
best_loss = float('inf')
counter = 0

# Training loop
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Move data to device
        anchor_input_ids = batch['anchor_input_ids'].to(device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(device)
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        negative_input_ids = batch['negative_input_ids'].to(device)
        negative_attention_mask = batch['negative_attention_mask'].to(device)

        # Compute embeddings
        anchor_embeddings = compute_embeddings(anchor_input_ids, anchor_attention_mask)
        positive_embeddings = compute_embeddings(positive_input_ids, positive_attention_mask)
        negative_embeddings = compute_embeddings(negative_input_ids, negative_attention_mask)

        # Normalize embeddings
        anchor_embeddings = nn.functional.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = nn.functional.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = nn.functional.normalize(negative_embeddings, p=2, dim=1)

        # Combine positive and negative embeddings
        embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)

        # Compute similarity scores
        logits = torch.matmul(anchor_embeddings, embeddings.T) / config.temperature

        # Create labels: positives are at positions [0, batch_size - 1]
        labels = torch.arange(anchor_embeddings.size(0), dtype=torch.long).to(device)

        # Compute loss
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        wandb.log({'train_loss': loss.item()})

    avg_train_loss = total_loss / len(train_dataloader)
    wandb.log({'avg_train_loss': avg_train_loss})

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            # Move data to device
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)

            # Compute embeddings
            anchor_embeddings = compute_embeddings(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = compute_embeddings(positive_input_ids, positive_attention_mask)
            negative_embeddings = compute_embeddings(negative_input_ids, negative_attention_mask)

            # Normalize embeddings
            anchor_embeddings = nn.functional.normalize(anchor_embeddings, p=2, dim=1)
            positive_embeddings = nn.functional.normalize(positive_embeddings, p=2, dim=1)
            negative_embeddings = nn.functional.normalize(negative_embeddings, p=2, dim=1)

            # Combine positive and negative embeddings
            embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)

            # Compute similarity scores
            logits = torch.matmul(anchor_embeddings, embeddings.T) / config.temperature

            # Create labels
            labels = torch.arange(anchor_embeddings.size(0), dtype=torch.long).to(device)

            # Compute loss
            val_loss = criterion(logits, labels)

            total_val_loss += val_loss.item()
            wandb.log({'val_loss': val_loss.item()})

    avg_val_loss = total_val_loss / len(test_dataloader)
    wandb.log({'avg_val_loss': avg_val_loss})

    print(f"Epoch {epoch+1}/{config.num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        # Save the best model
        model.save_pretrained('finetuned_model')
        tokenizer.save_pretrained('finetuned_model')
    else:
        counter += 1
        if counter >= config.patience:
            print("Early stopping triggered")
            break

# Save the final model
model.save_pretrained('finetuned_model')
tokenizer.save_pretrained('finetuned_model')
