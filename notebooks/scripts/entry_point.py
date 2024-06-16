# flake8: noqa: F401
import os
import sys

# import training function
from training import parse_args, train_fn



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)
