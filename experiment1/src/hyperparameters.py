import argparse
from datetime import datetime

def set_hyperparameters(parser):
    parser.add_argument("--parallel", action='store_true', help="Whether to train using multiple GPUs.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs through the dataset. Default=10")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size during training. Default=128")
    parser.add_argument("--N", type=int, default=6, help="Number of transformer layers. Default=6")
    parser.add_argument("--d-model", type=int, default=256, help="Dimension of the desired model. Default=256")
    parser.add_argument("--d-ff", type=int, default=1024, help="Dimension of the feedforward network. Default=1024")
    parser.add_argument("--h", type=int, default=8, help="Number of attention heads. Default=8")
    parser.add_argument("--image-layers", type=int, default=4, help="Number of image layers. Default=4")
    parser.add_argument("--activation", type=str, default="nn.ReLU", help="Activation function to use. Default=nn.ReLU")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate. Default=0.1")
    parser.add_argument("--warmup", type=int, default=2000, help="Optimizer warmup iterations. Default=2000")
    parser.add_argument("--min-freq-vocab", type=int, default=5, help="Minimum frequency of vocab to keep. Default=5")
    parser.add_argument("--max-len", type=int, default=50, help="Maxinum number of words per line to keep. Default=50")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluate the model every _ lines. Default=100")
    parser.add_argument("--cp-every", type=int, default=100000, help="Checkpoint the model every _ lines. Default=100,000")
