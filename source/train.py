import os
from tqdm import tqdm
import torch
from torch.optim import Adam
from transformers import AutoConfig

from models import MABELBert, training_objective
from dataset import CDADataset, get_dataloaders
from config import (
    ModelArguments, TrainingArguments, DEVICE, DATASET_ROOT, CKPT_DIR
)


def train_model(
        model,
        optimizer,
        criterion,
        train_dataloader,
        val_dataloader,
        num_epochs=2,
        lr_scheduler=None
):
    """
    Function to train a given nn.Module model
    :param model: nn.Module
    :param optimizer: torch.optim.SGD or similar
    :param criterion: nn.CrossEntropyLoss or similar
    :param train_dataloader: torch.utils.data.DataLoader
    :param val_dataloader: torch.utils.data.DataLoader
    :param num_epochs: int
    :param lr_scheduler: torch.optim.lr_scheduler.StepLR or None or similar
    :return: tuple(model, metrics)
    """

    model = model.to(DEVICE)

    # Training loop
    for epoch in range(num_epochs):
        metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0
        }

        # Training loop
        for i, input_dict in enumerate(tqdm(train_dataloader)):
            model.train()
            # Zero optim gradients
            optimizer.zero_grad()

            # Move to GPU
            for key, val in input_dict.items():
                input_dict[key] = val.to(DEVICE)

            # Forward pass
            outputs = model(**input_dict)
            loss = criterion(*outputs, input_dict["mlm_labels"])
            loss.backward()
            optimizer.step()

            # Calculate the loss
            metrics['train_loss'] += loss

        metrics['train_loss'] /= len(train_dataloader)

        # Validation loop
        # for i, ((X, case, lengths), y) in enumerate(tqdm(val_dataloader)):
        #     model.eval()
        #
        #     # Move to GPU
        #     X = X.to(DEVICE)  # (batch_size, seq_len)
        #     y = y.to(DEVICE)  # (batch_size, num_cls)
        #     case = case.to(DEVICE)
        #
        #     # Forward pass
        #     outputs = model(X, case, lengths)  # (batch_size, seq_len, num_cls)
        #     outputs = outputs.permute(0, 2, 1)  # (batch_size, num_cls, seq_len)
        #
        #     # Calculate the accuracy
        #     metrics['val_acc'] += \
        #         (torch.argmax(outputs, axis=1) == y).float().sum() / sum(lengths)
        #
        #     # Calculate the loss
        #     metrics['val_loss'] += loss

        # if lr_scheduler is not None:
        #     if isinstance(lr_scheduler, ReduceLROnPlateau):
        #         lr_scheduler.step(metrics['val_loss'])
        #     else:
        #         lr_scheduler.step()

        # metrics['val_loss'] /= len(val_dataloader)

        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print("Mode\tLoss")
        print(f"Train\t{metrics['train_loss']:.2f}")
        # print(f"Valid\t{metrics['val_loss']:.2f}\t{metrics['val_acc']:.2f}")

    return model


class Trainer:
    def __init__(self):

        pass

    def _get_dataloaders(self):
        pass

    def train(self):
        pass


def train(data_path, model_path):
    hf_config = AutoConfig.from_pretrained(ModelArguments.hf_model)

    # Setup dataset and dataloader
    data_path = os.path.join(DATASET_ROOT, data_path)
    dataset = CDADataset(data_path)
    train_dataloader = get_dataloaders(dataset)

    # Define the model and optimizer
    model = MABELBert(hf_config)
    optimizer = Adam(model.parameters(), TrainingArguments.lr)

    # Training loop
    model = train_model(
        model=model,
        optimizer=optimizer,
        criterion=training_objective,
        train_dataloader=train_dataloader,
        val_dataloader=None
    )

    torch.save(model, model_path)
    return model


if __name__ == '__main__':
    train('data_aug.csv', 'mabel.pth')
