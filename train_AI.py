import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import time

# Load the dataset
df = pd.read_csv("C:/Users/adiat/Downloads/Adya/Artificial intelligence/Final/Artificial-Intelligence/cleaned_data_AI/tokenized_with_embeddings.csv")
df = df.sample(n=4000, random_state=42)

# Tokenize raw text inputs
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")

# Tokenize the text data with padding and truncation
#inputs = tokenizer(
#    list(df['title'] + " " + df['content']),
#    return_tensors="pt",
#    padding=True,
#    truncation=True,
#    max_length=512
#)

# Tokenize the text data with padding and truncation
inputs = tokenizer(
    list(df['content']),  # Use the 'title' column
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

# Convert labels to tensors
labels = torch.tensor(df['status'].tolist(), dtype=torch.long)  

# Dataset and DataLoader setup
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the IndoBERT model
model = BertForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p2", 
    num_labels=len(df['status'].unique())
)
model.train()  # Set model to training mode

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 5
for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    epoch_start_time = time.time() 
    
    # Use tqdm for progress bar
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
        inputs_batch, attention_mask_batch, labels_batch = batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=inputs_batch,
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
        
        # Track loss for this epoch
        epoch_loss += loss.item()
    
    # Calculate and print the accuracy and average loss for this epoch
    epoch_accuracy = correct_predictions / total_predictions * 100
    epoch_loss_avg = epoch_loss / len(train_loader)
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} - Loss: {epoch_loss_avg:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s")
    
    # Save the model at the end of the epoch
    model_save_path = f"indobert_trained_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for Epoch {epoch + 1} saved to {model_save_path}")

# Final message
print("Training complete!")
