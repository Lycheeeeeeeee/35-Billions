from pathlib import Path
from train_helper import *

def main():
    # Train according to the data-dictionary #
    dictionary_path = "../data/fundtap-data-dictionary.csv"
    hyperparameters = ""
    model_dir = Path('../data/model_output')
    model_dir.mkdir(exist_ok=True, parents=True)
    new_customer = False
    processed_data_path = "../data/train.csv"
    # Run the training function
    train_profit_loss_binary(processed_data_path, dictionary_path, hyperparameters, model_dir, new_customer)

