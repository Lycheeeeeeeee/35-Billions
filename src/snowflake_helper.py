# from datetime import datetime
# from snowflake.snowpark import Session
# from snowflake.connector.pandas_tools import write_pandas
# import pandas as pd
# import json

# def create_snowflake_session(secret_file_path):
#     with open(secret_file_path, 'r') as file:
#         connection_parameters = json.load(file)
#     session = Session.builder.configs(connection_parameters).create()
#     return session

# def execute_sql(session, sql_query):
#     result = session.sql(sql_query).collect()
#     df = pd.DataFrame([row.as_dict() for row in result])
#     return df

# def upload_to_snowflake(session, df, table_name):


#     # Ensure all columns are properly formatted
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col] = df[col].astype(str)
#         elif pd.api.types.is_numeric_dtype(df[col]):
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     result = session.write_pandas( 
#                                 df = df, 
#                                 table_name = table_name, 
#                                 schema='TRUSTED',
#                                 overwrite=True,
#                                 auto_create_table = True,
#                                 table_type = 'transient'
#                                 )
#     return result

# # preprocess_helper.py
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# def load_data_dictionary(file_path):
#     data_dictionary = pd.read_csv(file_path)
#     data_dictionary.columns = data_dictionary.columns.str.lower()
#     data_dictionary['col_name'] = data_dictionary['columns_cleaned'].str.lower()
#     return data_dictionary

# def clean_numeric_columns(df, data_dictionary):
#     numeric_columns = data_dictionary[data_dictionary['data_type'] == 'NUMBER']['col_name'].tolist()
#     for col in numeric_columns:
#         if col in df.columns:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].str.replace(',', '').str.replace('%', '').str.replace('$', '').astype(float, errors='ignore')
#         else:
#             print(f'Column {col} not found in the training data')
#     return df

# def apply_fill_method(df, data_dictionary):
#     print("Applying fill methods...")
#     for _, row in data_dictionary.iterrows():
#         col_name = row['col_name']
#         fill_method = row['fill_method']
        
#         if fill_method == 'constant':
#             if col_name in df.columns:
#                 fill_value = row['fill_value'] if pd.notna(row['fill_value']) else 0
#                 print(f"Filling missing values in {col_name} with {fill_value}")
#                 df[col_name] = df[col_name].fillna(fill_value)
#             else:
#                 print(f"Warning: Column {col_name} not found for fill method 'constant'")
#         elif fill_method == 'previous':
#             prev_col_name = col_name.replace('yr1', 'yr0')
#             if col_name in df.columns and prev_col_name in df.columns:
#                 print(f"Filling missing values in {col_name} with values from {prev_col_name}")
#                 df[col_name] = df[col_name].fillna(df[prev_col_name])
#             else:
#                 print(f"Warning: Column {col_name} or {prev_col_name} not found for fill method 'previous'")
#         # Add more fill methods as needed
#     return df

# def date_string_to_day(x, level):
#     x = pd.to_datetime(x, errors='coerce')
#     if level == 'month':
#         return x.dt.day
#     elif level == 'year':
#         return x.dt.dayofyear
#     return x

# def subset_train_data(df, data_dictionary):
#     # Select columns for training
#     columns_for_training = data_dictionary[data_dictionary['use_for_training'] == 'Y']['col_name'].tolist()
#     train_data_subset = df[columns_for_training]
#     return train_data_subset