import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class StandardDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        premise = str(self.data.iloc[index]['premise'])
        hypothesis = str(self.data.iloc[index]['hypothesis'])
        label = int(self.data.iloc[index]['label'])

        inputs = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True
        )

        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)

        # Create MLM labels
        mlm_labels = input_ids.clone()
        probability_matrix = torch.full(mlm_labels.shape, 0.15)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mlm_labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # Replace masked input tokens with tokenizer.mask_token_id
        input_ids[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(label, dtype=torch.long),
            'mlm_labels': mlm_labels
        }
