import argparse
from pathlib import Path
from file_helper import check_file_exists, ensure_dir_exists
from train_helper import train_profit_loss_binary



def main(dictionary_path, processed_data_path, model_output_path):
    # Check if input files exist
    check_file_exists(dictionary_path)
    check_file_exists(processed_data_path)

    # Ensure output directory exists
    ensure_dir_exists(model_output_path)

    # Train according to the data-dictionary
    hyperparameters = ""  # You might want to specify hyperparameters here
    model_dir = Path(model_output_path)
    new_customer = False
    
    # Run the training function
    train_profit_loss_binary(processed_data_path, dictionary_path, hyperparameters, model_dir, new_customer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train profit loss binary model")
    parser.add_argument("--dictionary", required=True, help="Path to the data dictionary CSV")
    parser.add_argument("--data", required=True, help="Path to the processed training data CSV")
    parser.add_argument("--output", required=True, help="Path to the model output directory")
    
    args = parser.parse_args()
    
    main(args.dictionary, args.data, args.output)