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
    Extracts spans that are shared between two sequences.
    """
    seq1 = [str(x.item()) for x in seq1]
    seq2 = [str(x.item()) for x in seq2]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # Each op is a tuple: (operation, seq1_start, seq1_end, seq2_start, seq2_end)
        if op[0] == 'equal':
            template1.extend(range(op[1], op[2]))
            template2.extend(range(op[3], op[4]))

    return template1, template2


def get_log_probability(
        masked_token_ids, token_ids, mask_idx, model, device
):
    """
    Returns the log probability of the masked token.
    """
    # Ensure tensors are on the correct device
    masked_token_ids = masked_token_ids.to(device)
    token_ids = token_ids.to(device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(masked_token_ids)
        logits = outputs.logits  # Shape: (1, seq_length, vocab_size)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get the log probability of the true token at the masked index
    target_id = token_ids[0, mask_idx]
    log_prob = log_probs[0, mask_idx, target_id]

    return log_prob.item()


def evaluate_crows_pairs(
        model,
        tokenizer,
        bias_type="gender",
        device="cpu"
):
    crows_pairs_dataset = CrowsPairs(tokenizer, bias_type)

    batch_size = 1
    dataloader = DataLoader(crows_pairs_dataset, batch_size=batch_size)

    tok_mask_token = tokenizer.mask_token
    tok_mask_id = tokenizer.convert_tokens_to_ids(tok_mask_token)

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
        for idx in range(num_mask_tokens):
            mask_idx1 = template1[idx]
            mask_idx2 = template2[idx]

            # Create masked versions of the sentences
            sent1_mask = sent1.clone().detach().to(device)
            sent2_mask = sent2.clone().detach().to(device)

            sent1_mask[0, mask_idx1] = tok_mask_id
            sent2_mask[0, mask_idx2] = tok_mask_id

            # Compute log probabilities
            score1 += get_log_probability(
                sent1_mask, sent1, mask_idx1, model, device
            )
            score2 += get_log_probability(
                sent2_mask, sent2, mask_idx2, model, device
            )

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

    metric_score = (stereo_count + antistereo_count) / N * 100 if N > 0 else 0.0
    stereo_score = stereo_count / stereo_total * 100 if stereo_total > 0 else 0.0
    antistereo_score = antistereo_count / antistereo_total * 100 if antistereo_total > 0 else 0.0
    neutral_score = neutral_count / N * 100 if N > 0 else 0.0

    print("Results")
    print(f"Metric Score for {bias_type} -\t{metric_score:.2f}")
    print(f"Stereotype Score for {bias_type} -\t{stereo_score:.2f}")
    print(f"Anti-stereotype Score for {bias_type} -\t{antistereo_score:.2f}")
    print(f"Neutral Score for {bias_type} -\t{neutral_score:.2f}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the MABEL fine-tuned model
    model_name = 'princeton-nlp/mabel-bert-base-uncased'
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    model.eval()

    evaluate_crows_pairs(model, tokenizer, bias_type="gender", device=device)

