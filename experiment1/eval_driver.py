from src.experiment import autoencoder_eval_exp
from src.hyperparameters import set_hyperparameters

import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help="Name of the test dataset.")
parser.add_argument("--base_dir", type=str, default="../../text_corpora/prepared/", help="Data base directory.")
parser.add_argument("--data_dir", type=str, default="eng/test/", help="Data directory.")
parser.add_argument("--model_dir", type=str, default="../../outputs/models/", help="Model directory.")
parser.add_argument("--map_dir", type=str, default="../../outputs/maps/", help="Map directory.")
parser.add_argument("--train_name", type=str)

set_hyperparameters(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    autoencoder_eval_exp(args)