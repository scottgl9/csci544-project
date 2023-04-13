from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging
from tqdm import tqdm
from config import LoggerConfig


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(LoggerConfig.level)
formatter = logging.Formatter(LoggerConfig.format)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


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


def generate_word_pairs(wordlist_1, wordlist_2):
    # Check if the input lists are of equal length
    assert len(wordlist_1) == len(wordlist_2), \
        "Both wordlists should have same length"

    logger.debug(f"Word list 1: {wordlist_1[:10]}")
    logger.debug(f"Word list 2: {wordlist_2[:10]}")

    num_pairs = len(wordlist_1)
    word_pairs = {}

    for i in range(num_pairs):
        word_pairs.__setitem__(wordlist_1[i], wordlist_2[i])
        word_pairs.__setitem__(wordlist_2[i], wordlist_1[i])

    logger.info(f'Created {num_pairs} word pairs')

    return word_pairs


def prepare_dataset(wl_paths, out_file, **kwargs):
    data_og = load_huggingface_dataset(
        dataset_name=kwargs.get('dataset_name'),
        split=kwargs.get('split', 'train'),
        to_pandas=kwargs.get('to_pandas', True)
    )
    logger.info('Loaded huggingface dataset')
    assert len(wl_paths) == 2, "Must provide exactly two word lists"

    wl_p1, wl_p2 = wl_paths

    # Read the word lists
    with open(wl_p1) as file:
        wl_1 = file.readlines()

    wl_1 = [l.rstrip() for l in wl_1]

    with open(wl_p2) as file:
        wl_2 = file.readlines()

    wl_2 = [l.rstrip() for l in wl_2]

    word_pairs = generate_word_pairs(wl_1, wl_2)

    # Apply the generate CDA function on the dataframe
    tqdm.pandas()
    data_og[['aug_sent0', 'aug_sent1', 'both']] = data_og.progress_apply(
        generate_cda_df, axis=1, result_type='expand', args=(word_pairs,)
    )

    # Write augmented data to CSV file
    data_og.to_csv(out_file)
    logger.info(f'Augmented dataset stored at {out_file}')

    return data_og


if __name__ == '__main__':
    with open('data/wordlist/male_word_file.txt') as file:
        wordlist_1 = file.readlines()

    wordlist_1 = [l.rstrip() for l in wordlist_1]

    with open('data/wordlist/female_word_file.txt') as file:
        wordlist_2 = file.readlines()

    wordlist_2 = [l.rstrip() for l in wordlist_2]

    generate_word_pairs(wordlist_1, wordlist_2)
