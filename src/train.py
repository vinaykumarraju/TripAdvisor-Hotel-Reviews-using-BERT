import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer,BertForSequenceClassification
import pandas as pd
from tqdm import tqdm

# Use Apple MPS for acceleration
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load preprocessed dataset
df = pd.read_pickle("../data/preprocessed_tripadvisor.pkl")

# Tokenizer & Model (Using BERT)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
torch.nn.init.xavier_uniform_(model.classifier.weight)
torch.nn.init.zeros_(model.classifier.bias)

model.to(device)

# Convert text to tokenized inputs
class TripadvisorDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# Convert text and labels
texts = df["Review"].tolist()
labels = df["Rating"].tolist()

dataset = TripadvisorDataset(texts, labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Optimizer and Loss Function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Enable Mixed Precision (Speeds up Training)
scaler = torch.cuda.amp.GradScaler()

# Training Loop (Optimized for Speed)
EPOCHS = 3  
model.train()

for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in loop:
        optimizer.zero_grad()

        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):  # Mixed precision
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

# Save the trained model
torch.save(model.state_dict(), "../models/tripadvisor_optimized.pth")
print("Training complete! Model saved.")
