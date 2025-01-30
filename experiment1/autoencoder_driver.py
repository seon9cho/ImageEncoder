from src.experiment import autoencoder_new_session, autoencoder_continue_session
from src.hyperparameters import set_hyperparameters

import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="toy_data.txt", help="Name of the data file.")
parser.add_argument("--val-name", type=str, help="Name of the validation data file.")
parser.add_argument("--base-dir", type=str, default="../../text_corpora/prepared/", help="Base directory.")
parser.add_argument("--data-dir", type=str, default="eng/", help="Data directory.")
parser.add_argument("--model-dir", type=str, default="../../outputs/models/", help="Model directory.")
parser.add_argument("--map-dir", type=str, default="../../outputs/maps/", help="Map directory.")
parser.add_argument("--continue-session", action='store_true', help="Continue training from an existing session.")
parser.add_argument("--train-name", type=str, help="Name of the existing training session.")
parser.add_argument("--check-point", type=int, default=1, help="Check point of existing model.")
set_hyperparameters(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.continue_session:
        autoencoder_continue_session(args)
    else:
        autoencoder_new_session(args)