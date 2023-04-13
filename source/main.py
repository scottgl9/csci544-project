import os
import sys
import logging
from argparse import ArgumentParser
from utils import prepare_dataset
from config import LoggerConfig, DATASET_ROOT


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(LoggerConfig.level)
formatter = logging.Formatter(LoggerConfig.format)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# Setup argument parser
def setup_cli():
    global_parser = ArgumentParser()

    # Subparser for dataset preprocessing
    subparsers = global_parser.add_subparsers(
        title='preprocess', help='Dataset preprocessing'
    )

    preprocess_parser = subparsers.add_parser('data_prep')
    preprocess_parser.add_argument('--dataset-name', '-d',
                                   default='snli',
                                   help='Name of the huggingface dataset',
                                   dest='dataset_name')
    preprocess_parser.add_argument('--out-file', '-o',
                                   default=f'{DATASET_ROOT}/data_aug.csv',
                                   help='Path of the output file',
                                   dest='out_file')
    preprocess_parser.add_argument('--wl-path',
                                   help='Path of the word lists. 2 required',
                                   action='append',
                                   dest='wl_paths')
    preprocess_parser.set_defaults(func=prepare_dataset)

    # TODO: Subparser for training

    # TODO: Subparser for evaluation

    # TODO: Subparser for inference

    return global_parser


if __name__ == '__main__':
    # args.func(**vars(args))
    cli = setup_cli()
    args = cli.parse_args()
    print(args)
