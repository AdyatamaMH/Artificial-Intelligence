import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW
from tqdm import tqdm
import time

# Load the dataset
df = pd.read_csv("C:/Users/adiat/Downloads/Adya/Artificial intelligence/Final/Artificial-Intelligence/cleaned_data_AI/tokenized_with_embeddings.csv")
df = df.sample(n=800, random_state=42)

# Extract pre-tokenized input IDs for title and content
title_ids = df['title_token_ids'].apply(eval).tolist()  
content_ids = df['content_token_ids'].apply(eval).tolist()

# Concatenate title and content IDs if desired
input_ids = [title + content for title, content in zip(title_ids, content_ids)]

# Define maximum length 
max_length = 512

# Pad the sequences to the same length 
input_ids = [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in input_ids]

# Create attention masks (1 for tokens, 0 for padding)
attention_masks = [[1 if token_id != 0 else 0 for token_id in ids] for ids in input_ids]

# Convert to tensors
input_ids = torch.tensor(input_ids, dtype=torch.long)
attention_masks = torch.tensor(attention_masks, dtype=torch.long)
labels = torch.tensor(df['status'].tolist(), dtype=torch.long)

# Dataset and DataLoader setup
dataset = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the IndoBERT model
model = BertForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2", 
    num_labels=len(df['status'].unique())
)
model.train() 

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 5
for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    epoch_start_time = time.time()
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
        input_ids_batch, attention_mask_batch, labels_batch = batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch,
            labels=labels_batch
        )
        loss = outputs.loss
        logits = outputs.logits
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions += (predictions == labels_batch).sum().item()
        total_predictions += labels_batch.size(0)
        
        # Track loss
        epoch_loss += loss.item()
    
    # Epoch metrics
    epoch_accuracy = correct_predictions / total_predictions * 100
    epoch_loss_avg = epoch_loss / len(train_loader)
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} - Loss: {epoch_loss_avg:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s")
    
    # Save the model
    model_save_path = f"indobert_trained_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for Epoch {epoch + 1} saved to {model_save_path}")

# Final message
print("Training complete!")
