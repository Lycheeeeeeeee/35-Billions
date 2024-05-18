from sklearn.model_selection import  train_test_split
from sklearn.metrics import  matthews_corrcoef, brier_score_loss
import pandas as pd
import numpy as np

def read_data(data_path):
    data = pd.read_csv(data_path)
    cat_features = list(data.select_dtypes(include=['object']).columns)
    data[cat_features] = data[cat_features].astype("category") 
    return data

def balanced_train_validation_test(data, target_column,random_state):
    X = data.loc[:, data.columns != target_column]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def fundtap_train_test_split(data,target_column = "label",
                     hold_out_columns = ["quote","fundtapprofitloss"],
                    random_state = 456):
    
    categorical_feature = list(data.select_dtypes(include=['object']).columns)
    data[categorical_feature] = data[categorical_feature].astype("category")
    X_train, X_test, y_train, y_test = balanced_train_validation_test(data.loc[:, ~data.columns.isin(hold_out_columns)],target_column,random_state)
    
    train_instance_weight = np.abs(data.loc[X_train.index]["fundtapprofitloss"])
    test_instance_weight = np.abs(data.loc[X_test.index]["fundtapprofitloss"])

    return X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight


