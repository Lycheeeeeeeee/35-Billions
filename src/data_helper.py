# data_helper #
from sklearn.model_selection import  train_test_split
import numpy as np
import pandas as pd
## load dictionary
def load_dictionary(dictionary_path):
    data_dictionary = pd.read_csv(dictionary_path, low_memory=False)
    columns_for_training = data_dictionary[data_dictionary["use_for_training"] == "Y"]["columns_cleaned"].tolist()
    hold_out_columns = data_dictionary[data_dictionary["hold_out_columns"] == "Y"]["columns_cleaned"].tolist()
    prediction_target = data_dictionary[data_dictionary["prediction_target"] == "Y"]["columns_cleaned"].tolist()
    return columns_for_training, hold_out_columns,prediction_target

## select features
def select_data(data, dictionary_path):
    columns_for_training, hold_out_columns,prediction_target = load_dictionary(dictionary_path)
    # this needs improvement - needs to be in data processing notebook? or upper stream.
    data["label"] = data[prediction_target] >= 0  

    target_column = "label"
    data = data[columns_for_training + [target_column]]

    # Define catgorical features #
    cat_features = list(data.select_dtypes(include=['object']).columns)
    data[cat_features] = data[cat_features].astype("category") 
    return data, hold_out_columns, target_column

## train vs. test split
def balanced_train_validation_test(data, target_column,random_state):
    X = data.loc[:, data.columns != target_column]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

## split function
def fundtap_train_test_split(data, dictionary_path,random_state=456):  
    data, hold_out_columns,target_column = select_data(data, dictionary_path)
    X_train, X_test, y_train, y_test = balanced_train_validation_test(data.loc[:, ~data.columns.isin(hold_out_columns)],
                                                                      target_column,random_state)
    
    train_instance_weight = np.abs(data.loc[X_train.index]["fundtap_loan_principal"]) 
    # fundtap_profit_loss? also needs to be added to dictionary
    test_instance_weight = np.abs(data.loc[X_test.index]["fundtap_loan_principal"])

    train_holdout = data.loc[X_train.index, hold_out_columns]
    test_holdout = data.loc[X_test.index, hold_out_columns]

    return X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight,train_holdout,test_holdout