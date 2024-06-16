# flake8: noqa: F401
import os
import sys

# import training function
from training import parse_args

# import deployment functions
from explaining import model_fn

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    model_fn(args.model_dir)
