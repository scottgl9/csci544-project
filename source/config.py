import os
import logging
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = os.path.dirname(__file__)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'out')


class ModelArguments:
    hf_model = 'bert-base-uncased'
    max_len = 128


class TrainingArguments:
    lr = 5e-5
    batch_size = 2
    mlm_prob = 0.15


class LoggerConfig:
    level = logging.DEBUG
    format = '[%(levelname)s][%(name)s:%(lineno)d] %(message)s'
