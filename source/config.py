import os
import logging
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = os.path.dirname(__file__)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data')


class LoggerConfig:
    level = logging.DEBUG
    format = '[%(levelname)s][%(name)s:%(lineno)d] %(message)s'
