import logging
from argparse import ArgumentParser
from utils import prepare_dataset, prepare_dataset_base
from train import train
from train_base import train_base
from evaluate import evaluate
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
        title="preprocess", help="Dataset preprocessing"
    )

    preprocess_parser = subparsers.add_parser("data_prep")
    preprocess_parser.add_argument("--dataset-name", "-d",
                                   default="snli",
                                   help="Name of the huggingface dataset",
                                   dest="dataset_name")
    preprocess_parser.add_argument("--out-file", "-o",
                                   default=f"{DATASET_ROOT}/data_aug.csv",
                                   help="Path of the output file",
                                   dest="out_file")
    preprocess_parser.add_argument("--wl-path",
                                   help="Path of the word lists. 2 required",
                                   action="append",
                                   dest="wl_paths")
    preprocess_parser.set_defaults(func=prepare_dataset)

    preprocess_parser = subparsers.add_parser("data_prep_base")
    preprocess_parser.add_argument("--dataset-name", "-d",
                                   default="snli",
                                   help="Name of the huggingface dataset",
                                   dest="dataset_name")
    preprocess_parser.add_argument("--out-file", "-o",
                                   default=f"{DATASET_ROOT}/data.csv",
                                   help="Path of the output file",
                                   dest="out_file")
    preprocess_parser.set_defaults(func=prepare_dataset_base)

    # Subparser for training
    training_parser = subparsers.add_parser("train")
    training_parser.add_argument("--data-path", "-d",
                                 help="Path to the counterfactually augmented "
                                      "dataset",
                                 dest="data_path")
    training_parser.add_argument("--ckpt-path", "-o",
                                 help="Path to store the trained model file",
                                 dest="model_path")
    training_parser.set_defaults(func=train)

    # Subparser for training
    training_parser = subparsers.add_parser("train_base")
    training_parser.add_argument("--data-path", "-d",
                                 help="Path to the counterfactually augmented "
                                      "dataset",
                                 dest="data_path")
    training_parser.add_argument("--ckpt-path", "-o",
                                 help="Path to store the trained model file",
                                 dest="model_path")
    training_parser.set_defaults(func=train_base)

    # Subparser for evaluation
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--ckpt-path", "-i",
                             help="Path to the stored model",
                             dest="ckpt_path")
    eval_parser.add_argument("--dataset-name", "-d",
                             help="Intrinsic metric evaluation. "
                                  "Choose from [stereoset, crows]",
                             dest="dataset")
    eval_parser.set_defaults(func=evaluate)

    return global_parser


if __name__ == "__main__":
    cli = setup_cli()
    args = cli.parse_args()
    args.func(**vars(args))
