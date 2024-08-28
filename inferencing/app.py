import os
import sys
import json
import logging
from typing import Dict, Any, List

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocessing_helper import load_data_dictionary, clean_numeric_columns, apply_fill_method, subset_train_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rest of your app.py code remains the same...

# Initialize FastAPI app
app = FastAPI()

# Load model
MODEL_FILE = os.getenv('MODEL_FILE', 'model/your_model.joblib')
if not os.path.exists(MODEL_FILE):
    raise ValueError(f"Model file not found: {MODEL_FILE}")

try:
    model = joblib.load(MODEL_FILE)
    logger.info(f"Model loaded successfully from {MODEL_FILE}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load data dictionary
DATA_DICTIONARY_PATH = os.getenv('DATA_DICTIONARY_PATH', 'data/fundtap-data-dictionary.csv')
data_dict = load_data_dictionary(DATA_DICTIONARY_PATH)

class PredictionInput(BaseModel):
    features: Dict[str, Any]

class PredictionOutput(BaseModel):
    profitability_probability: float
    predicted_profit_loss_label: int
    warning_message: str

def preprocess_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    # Load data dictionary
    data_dict = load_data_dictionary(DATA_DICTIONARY_PATH)
    df.columns = df.columns.str.lower()

    # Clean numeric columns
    df = clean_numeric_columns(df, data_dict)

    # Apply fill methods for missing values
    df = apply_fill_method(df, data_dict)

    # Subset data for training
    df_subset = subset_train_data(df, data_dict, False)
    # Ensure the DataFrame only includes columns from the data dictionary

    # Define categorical features
    cat_features = data_dict[data_dict['data_type'] == 'VARCHAR']['col_name'].tolist()
    cat_features = list(set(cat_features) & set(df_subset.columns))
    for feature in cat_features:
        if df_subset[feature].dtype == 'object':
            df_subset[feature] = df_subset[feature].astype('category')
        else:
            logger.warning(f"Feature {feature} is marked as VARCHAR in data dictionary but is not of object type in the DataFrame. Current type: {df_subset[feature].dtype}")

    # Define numerical features
    cat_features = data_dict[data_dict['data_type'] == 'NUMBER']['col_name'].tolist()
    cat_features = list(set(cat_features) & set(df_subset.columns))
    for feature in cat_features:
        if df_subset[feature].dtype == 'object':
            df_subset[feature] = 0
            logger.warning(f"Feature {feature} is marked as NUMBER in data dictionary but is not of object type in the DataFrame. Current type: {df_subset[feature].dtype}. Converting to 0")

    # Log the data types of all columns
    logger.info("Column data types:")
    for col in df_subset.columns:
        logger.info(f"{col}: {df_subset[col].dtype}")
    
    return df_subset

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput) -> Dict[str, Any]:
    try:
        # Preprocess input data
        features = preprocess_input(input_data.features)
        print(features.dtypes)
        # Make prediction
        y_pred = model.predict(features)
        
        # Assuming the model output is a single value between 0 and 1
        probability = float(y_pred[0])
        
        # Determine profit loss label (assuming 0.5 as threshold)
        profit_loss_label = int(probability > 0.5)
        
        # Generate warning message (implement your logic here)
        warning_message = "No warnings"  # Placeholder
        
        return {
            "profitability_probability": probability,
            "predicted_profit_loss_label": profit_loss_label,
            "warning_message": warning_message
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info() -> Dict[str, Any]:
    return {
        "model_file": MODEL_FILE,
        "data_dictionary": DATA_DICTIONARY_PATH,
        "feature_names": list(data_dict['column_name'])
    }

# Lambda handler
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        # Parse the incoming event
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        
        # Create a PredictionInput instance
        input_data = PredictionInput(features=body)
        
        # Call the prediction function
        result = predict(input_data)
        
        return {
            'isBase64Encoded': False,
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        logger.error(f"Lambda handler error: {e}")
        return {
            'isBase64Encoded': False,
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)