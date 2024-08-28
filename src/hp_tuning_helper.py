
# hp_tuning_helper #
import json
from pathlib import Path

def get_warm_start_parameter(model_output_path, new_customer):
    """
    Checks if the warm start parameter JSON file exists.
    If it exists, reads and returns the parameters.
    If it does not exist, returns an empty string.

    Args:
    model_dir (Path): Path to the model directory.
    new_customer (bool): Flag indicating if it's a new customer.

    Returns:
    dict or str: Parameters from the file if it exists, otherwise an empty string.
    """
    if new_customer:
        warm_start_file = 'new_customer_profitloss_hyperparameter.json'
    else:
        warm_start_file = 'existing_customer_profitloss_hyperparameter.json'
    
    model_dir = Path(model_output_path)
    full_path = model_dir / warm_start_file
    
    if full_path.exists():
        with open(full_path, 'r') as file:
            parameters = json.load(file)
            return parameters
    else:
        return ""