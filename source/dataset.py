from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

from config import ModelArguments, TrainingArguments


class CDADataset(Dataset):
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.data = self.data.iloc[:2000]
        self.tokenizer = AutoTokenizer.from_pretrained(ModelArguments.hf_model)
        self.columns = [
            "orig_sent0",
            "orig_sent1",
            "aug_sent0",
            "aug_sent1",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        datum = self.data.loc[item]
        features = {'both': datum['both']}

        for col in self.columns:
            features[col] = self.tokenizer(
                datum[col],
                truncation=True,
                max_length=ModelArguments.max_len,
                padding="max_length"
            )

        return features


@dataclass
class MLMDataCollator:
    tokenizer = AutoTokenizer.from_pretrained(ModelArguments.hf_model)

    def collate_fn(self, batch):
        """
        Collate function to create a masked language dataset
        :param batch:
        :return:
        """
        self.batch_size = len(batch)
        self.nums_sent = 4  # Number of sentences in each example. Orig (2), Aug (2)
        collate_batch = {
            'input_ids': torch.zeros((self.batch_size, self.nums_sent, ModelArguments.max_len), dtype=torch.long),
            'attention_mask': torch.zeros((self.batch_size, self.nums_sent, ModelArguments.max_len), dtype=torch.long),
            'token_type_ids': torch.zeros((self.batch_size, self.nums_sent, ModelArguments.max_len), dtype=torch.long),
        }

        columns = [
            "orig_sent0",
            "orig_sent1",
            "aug_sent0",
            "aug_sent1",
        ]

        for i, example in enumerate(batch):
            for col_num, col in enumerate(columns):
                for key in collate_batch.keys():
                    collate_batch[key][i, col_num, :] = torch.tensor(example[col][key], dtype=torch.long)

        collate_batch["mlm_input_ids"], collate_batch["mlm_labels"] = self.mask_inputs(collate_batch["input_ids"].view(-1, ModelArguments.max_len))
        return collate_batch

    def mask_inputs(self, inputs: torch.Tensor):
        inputs = inputs.clone()
        labels = inputs.clone()

        probability_matrix = torch.full(inputs.shape, TrainingArguments.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True
            )
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask,
                                           dtype=torch.bool)

        # Mask only non-special tokens
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Compute loss only on masked tokens

        # Masking logic
        # 80% - Mask the word
        # 10% - Replace the actual word with a random word
        # 10% - Do nothing

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # return (inputs.view(self.batch_size, self.nums_sent, -1),
        #         labels.view(self.batch_size, self.nums_sent, -1))
        return inputs, labels


def get_dataloaders(train_dataset, **kwargs):
    collator = MLMDataCollator()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=TrainingArguments.batch_size,
        shuffle=True,
        collate_fn=collator.collate_fn
    )
    return train_dataloader


if __name__ == '__main__':
    dataset = CDADataset('./data/data_aug.csv')
    dataloader = get_dataloaders(dataset)
    next(iter(dataloader))
    print(dataset[0])
