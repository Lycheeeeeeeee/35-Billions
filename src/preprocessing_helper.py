# preprocess_helper.py
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def load_data_dictionary(file_path):
    data_dictionary = pd.read_csv(file_path)
    data_dictionary.columns = data_dictionary.columns.str.lower()
    data_dictionary['col_name'] = data_dictionary['columns_cleaned'].str.lower()
    return data_dictionary

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_numeric_columns(df, data_dictionary):
    numeric_columns = data_dictionary[data_dictionary['data_type'] == 'NUMBER']['col_name'].tolist()
    for col in numeric_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                # Replace empty strings with NaN
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
                
                # Clean the column
                df[col] = df[col].str.replace(',', '').str.replace('%', '').str.replace('$', '')
                
                # Convert to float, replacing NaN with 0
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            elif pd.api.types.is_numeric_dtype(df[col]):
                # For already numeric columns, replace NaN with 0
                df[col] = df[col].fillna(0)
            
            logger.info(f"Cleaned column {col}. Sample values: {df[col].head()}")
        else:
            logger.warning(f'Column {col} not found in the training data')
    
    return df

def apply_fill_method(df, data_dictionary):
    print("Applying fill methods...")
    for _, row in data_dictionary.iterrows():
        col_name = row['col_name']
        fill_method = row['fill_method']
        
        if fill_method == 'constant':
            if col_name in df.columns:
                fill_value = row['fill_value'] if pd.notna(row['fill_value']) else 0
                print(f"Filling missing values in {col_name} with {fill_value}")
                df[col_name] = fill_value
            else:
                print(f"Warning: Column {col_name} not found for fill method 'constant'")
        elif fill_method == 'previous':
            prev_col_name = fill_method = row['fill_value']
            if col_name in df.columns and prev_col_name in df.columns:
                print(f"Filling missing values in {col_name} with values from {prev_col_name}")
                df[col_name] = df[col_name].fillna(df[prev_col_name])
            else:
                print(f"Warning: Column {col_name} or {prev_col_name} not found for fill method 'previous'")
        # Add more fill methods as needed
    return df

def date_string_to_day(x, level):
    x = pd.to_datetime(x, errors='coerce')
    if level == 'month':
        return x.dt.day
    elif level == 'year':
        return x.dt.dayofyear
    return x

def subset_train_data(df, data_dictionary, include_hold_out=True):
    if include_hold_out:
        columns_for_training = data_dictionary[data_dictionary['use_for_training'] == 'Y']['col_name'].tolist()
    else:
        columns_for_training = data_dictionary[(data_dictionary['use_for_training'] == 'Y') & 
                                               (data_dictionary['hold_out_columns'] != 'Y')]['col_name'].tolist()
    train_data_subset = df[columns_for_training]
    return train_data_subset
