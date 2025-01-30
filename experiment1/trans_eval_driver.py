from src.experiment import semisup_eval_exp, sup_eval_exp
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
parser.add_argument("--supervised", action="store_true")
parser.add_argument("--base_dir", type=str, default="../../text_corpora/prepared/", help="Data base directory.")
parser.add_argument("--data_dir", type=str, default="fra-eng/test/", help="Data directory.")
parser.add_argument("--model_dir", type=str, default="../../outputs/models/", help="Model directory.")
parser.add_argument("--map_dir", type=str, default="../../outputs/maps/", help="Model directory.")
parser.add_argument("--train_name", type=str)
parser.add_argument("--src_model_name", type=str, help="Source language model train name.")
parser.add_argument("--trg_model_name", type=str, help="Target language model train name.")
parser.add_argument("--src_map_name", type=str, help="Source language model train name.")
parser.add_argument("--trg_map_name", type=str, help="Target language model train name.")

set_hyperparameters(parser)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.supervised:
        sup_eval_exp(args)
    else:
        semisup_eval_exp(args)