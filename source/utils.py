from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def load_huggingface_dataset(dataset_name, split='train', to_pandas=True):
    dataset = load_dataset(dataset_name, split=split)
    if to_pandas:
        return dataset.to_pandas()
    else:
        return dataset


def generate_cda(sentence, vocab_pairs):
    sentence = sentence.split(' ')
    transformed = False
    for i in range(len(sentence)):
        if sentence[i] in vocab_pairs:
            sentence[i] = vocab_pairs[sentence[i]]
            transformed = True
    return ' '.join(sentence), transformed


def generate_cda_df(df, vocab_pairs):
    aug_sent0, status0 = generate_cda(df.premise, vocab_pairs)
    aug_sent1, status1 = generate_cda(df.hypothesis, vocab_pairs)

    return aug_sent0, aug_sent1, int(status0 and status1)
