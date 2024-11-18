import os
import torch
from transformers import BertForPreTraining, AdamW
from dataset_base import StandardDataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from config import (
    ModelArguments, TrainingArguments, DEVICE, DATASET_ROOT, CKPT_DIR
)

def get_dataloaders(data_path, batch_size, max_len):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = StandardDataset(data_path, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class Trainer:
    def __init__(self):
        pass

    def train(self, data_path, model_path, batch_size=64, max_len=128, epochs=1, lr=TrainingArguments.lr):
        # Setup dataset and dataloader
        train_dataloader = get_dataloaders(data_path, batch_size, max_len)

        # Define the model and optimizer
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        model.to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        print("Training model...")

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                mlm_labels = batch['mlm_labels']
                labels = batch['labels']

                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=mlm_labels, next_sentence_label=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs} completed with average loss: {avg_loss}")

        #torch.save(model.state_dict(), model_path)
        torch.save(model, model_path)
        return model

def train_base(data_path, model_path, **kwargs):
    trainer = Trainer()
    data_path = os.path.join('data', data_path)
    return trainer.train(data_path, model_path)#, **kwargs)

if __name__ == '__main__':
    train_base('data.csv', 'base.pth')