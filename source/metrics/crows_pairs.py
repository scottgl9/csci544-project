import string
from tqdm import tqdm
import numpy as np
import difflib

from datasets import load_dataset
import transformers

import torch
from torch.utils.data import DataLoader, Dataset


class CrowsPairs(Dataset):
    def __init__(self, tokenizer, bias_type="gender"):
        self.tokenizer = tokenizer
        self.crows_pairs = load_dataset("crows_pairs", split="test")
        bias_label_map = {
            "race-color": 0, "socioeconomic": 1, "gender": 2, "disability": 3,
            "nationality": 4, "sexual-orientation": 5, "physical-appearance": 6,
            "religion": 7, "age": 8
        }
        self.bias_type = bias_label_map[bias_type]
        self.data = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for sample in self.crows_pairs:
            if sample["bias_type"] == self.bias_type:
                data_sample = (
                    self.tokenizer.encode(sample["sent_more"],
                                          return_tensors='pt'),
                    self.tokenizer.encode(sample["sent_less"],
                                          return_tensors='pt'),
                    sample["stereo_antistereo"]
                )
                self.data.append(data_sample)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    Taken from https://github.com/nyu-mll/crows-pairs/blob/8aaac11c485473159ec9328a65253a5be9a479dc/metric.py#L126
    """
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def get_log_probability(
        masked_token_ids, token_ids, mask_idx, model
):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    # get model hidden states
    output = model(masked_token_ids)
    # print(output)
    hidden_states = output[0].squeeze(0)
    # mask_id = tokenizer.convert_tokens_to_ids(tok_mask_token)

    # # we only need log_prob for the MASK tokens
    # assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = torch.nn.LogSoftmax(dim=0)(hs)[target_id]

    return log_probs.item()


def evaluate_crows_pairs(
        model,
        bias_type="gender",
        device="cpu"
):
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "bert-base-cased", padding_side='right'
    )
    crows_pairs = CrowsPairs(tokenizer, bias_type)

    batch_size = 1
    dataloader = DataLoader(crows_pairs, batch_size=batch_size)

    tok_mask_token = tokenizer.mask_token
    tok_mask_id = tokenizer.encode(tok_mask_token, add_special_tokens=False)[0]

    neutral_count = 0
    stereo_count = 0
    antistereo_count = 0
    stereo_total = 0
    antistereo_total = 0

    N = len(dataloader)
    for sample in tqdm(dataloader):
        sent1, sent2, direction = sample
        sent1 = sent1.squeeze(0)
        sent2 = sent2.squeeze(0)

        direction = direction.item()

        template1, template2 = get_span(sent1[0], sent2[0])

        num_mask_tokens = len(template1)

        score1, score2 = 0, 0
        for mask_idx in range(1, num_mask_tokens - 1):
            # print(mask_idx)
            sent1_mask = sent1.clone().detach()
            sent2_mask = sent2.clone().detach()

            sent1_mask[0][template1[mask_idx]] = tok_mask_id
            sent2_mask[0][template2[mask_idx]] = tok_mask_id

            score1 += get_log_probability(
                sent1_mask, sent1, template1[mask_idx], model
            )
            score2 += get_log_probability(
                sent2_mask, sent2, template2[mask_idx], model
            )

        # print(score1, score2)
        if score1 == score2:
            neutral_count += 1
        else:
            if direction == 0:
                stereo_total += 1
                if score1 > score2:
                    stereo_count += 1
            elif direction == 1:
                antistereo_total += 1
                if score2 > score1:
                    antistereo_count += 1

    metric_score = (stereo_count + antistereo_count) / N * 100
    stereo_score = stereo_count / stereo_total * 100
    antistereo_score = antistereo_count / antistereo_total * 100
    neutral_score = neutral_count / N * 100

    print("Results")
    print(f"Metric Score for {bias_type} -\t{metric_score}")
    print(f"Stereotype Score for {bias_type} -\t{stereo_score}")
    print(f"Anti-stereotype Score for {bias_type} -\t{antistereo_score}")


if __name__ == '__main__':
    model = transformers.BertForMaskedLM.from_pretrained('bert-base-cased')

    evaluate_crows_pairs(model, "nationality")
