from src.experiment import train_translation_supervised
from src.hyperparameters import set_hyperparameters

import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--src_filename", type=str, 
    default="fra.txt",
    help="Name of the source language file.")
parser.add_argument("--trg_filename", type=str, 
    default="eng.txt",
    help="Name of the target language file.")
parser.add_argument("--train", type=bool, default=True, help="Whether to train a new model or evaluate an existing model. Default=True")
parser.add_argument("--data_dir", type=str, default="../../text_corpora/prepared/fra-eng/", help="data directory.")

set_hyperparameters(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.train:
        train_translation_supervised(args)