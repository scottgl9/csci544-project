import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm import tqdm
import os

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./bert_snli_model.pth"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

# Load SNLI dataset
print("Loading SNLI dataset...")
snli = load_dataset("snli")

# only train on examples where label is 0 (entailment)
snli["train"] = snli["train"].filter(lambda example: example["label"] == 0)
snli["validation"] = snli["validation"].filter(lambda example: example["label"] == 0)

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

# Tokenize datasets
print("Tokenizing the datasets...")
train_dataset = snli["train"].map(preprocess_function, batched=True)
val_dataset = snli["validation"].map(preprocess_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["premise", "hypothesis"])
val_dataset = val_dataset.remove_columns(["premise", "hypothesis"])
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss / len(train_loader):.4f}")
    print(f"Train Accuracy: {100 * correct / total:.2f}%")

    # Validation step
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
print(f"Saving the trained model to {OUTPUT_DIR}...")
torch.save(model, OUTPUT_DIR)
print("Model saved!")
