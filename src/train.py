import argparse
from pathlib import Path
from file_helper import check_file_exists, ensure_dir_exists
from train_helper import train_profit_loss_binary_on_weighted_mcc,train_profit_loss_binary_on_log_loss

def main(dictionary_path, processed_data_path, model_output_path, new_customer, n_trials):
    # Check if input files exist
    check_file_exists(dictionary_path)
    check_file_exists(processed_data_path)

    # Ensure output directory exists
    ensure_dir_exists(model_output_path)

    # Train according to the data-dictionary
    model_dir = Path(model_output_path)
    
    # Run the training function
    train_profit_loss_binary_on_weighted_mcc(processed_data_path, dictionary_path, model_dir, new_customer, n_trials)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train profit loss binary model")
    parser.add_argument("--dictionary", required=True, help="Path to the data dictionary CSV")
    parser.add_argument("--data", required=True, help="Path to the processed training data CSV")
    parser.add_argument("--output", required=True, help="Path to the model output directory")
    
    # (action="store_true"), which will be False by default unless specified.
    parser.add_argument("--new-customer", action="store_true", help="Flag for new customer (default: False)")

    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials to run (default: 10)")
    
    
    args = parser.parse_args()
    
    main(args.dictionary, args.data, args.output, args.new_customer, args.n_trials)