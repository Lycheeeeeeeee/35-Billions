#memmory_helper

import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Call this function at key points in your script

# training_helper #
from sklearn.model_selection import  train_test_split

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
    
    train_instance_weight = np.abs(data.loc[X_train.index]["fundtap_profit_loss"])
    test_instance_weight = np.abs(data.loc[X_test.index]["fundtap_profit_loss"])

    train_holdout = data.loc[X_train.index, hold_out_columns]
    test_holdout = data.loc[X_test.index, hold_out_columns]

    return X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight,train_holdout,test_holdout

# hp_tuning_helper #
import optuna
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb

def learning_rate_decay_power_0995(current_iter): 
    base_learning_rate = 0.1 
    lr = base_learning_rate * np.power(.995, current_iter) 
    return lr if lr > 1e-3 else 1e-3


def hp_tuning_init_param(X_train, y_train, metric, k_folds, feval,random_state):
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    dtrain = opt_lgb.Dataset(X_train, label=y_train)
    
    fixed_param =  params = {
        "objective": "binary",
        "metric": metric,
        "boosting_type": "gbdt",
        'random_state' : random_state,
        'verbose':-1,
        'is_unbalance': 'true'
    }

    tuner = opt_lgb.LightGBMTunerCV(
        fixed_param, dtrain, verbose_eval=False,
        early_stopping_rounds=100, 
        nfold = k_folds,
        stratified = True,
        show_progress_bar = False,
        optuna_seed = random_state,
        feval=feval,
        callbacks=[lgb.reset_parameter(learning_rate = learning_rate_decay_power_0995) ]
    )

    

    tuner.run()
    return tuner.best_params 


# result_interpretation_helper #
from sklearn.metrics import log_loss, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, brier_score_loss
import pandas as pd
import numpy as np

def accuracy_analysis(classifier, X_train, X_test, y_train, y_test, test_instance_weight ):

        pred_proba = classifier.predict(X_test)
        yhat =np.where(pred_proba < 0.5, 0, 1) 

        mcc = matthews_corrcoef(y_test, yhat)
        logloss = log_loss(y_test, pred_proba)
        bs= brier_score_loss(y_test, pred_proba)


        weighted_mcc = matthews_corrcoef(y_test, yhat, sample_weight = test_instance_weight)
        weighted_logloss = log_loss(y_test, pred_proba,sample_weight=test_instance_weight)
        weighted_bs= brier_score_loss(y_test, pred_proba, sample_weight= test_instance_weight)


        tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
        fpr = (fp)/ float(fp + tn)
        fnr = (fn)/ float(fn + tp)
        train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, np.where(classifier.predict(X_train) < 0.5, 0, 1) ).ravel()
        train_fpr = (train_fp)/ float(train_fp + train_tn)
        train_fnr = (train_fn)/ float(train_fn + train_tp)

        accuracy_df = pd.DataFrame([  weighted_mcc, weighted_logloss, weighted_bs, mcc, logloss, bs, tn, fp, fn, tp, fpr, fnr, train_tn, train_fp, train_fn, train_tp,train_fpr,train_fnr])
        accuracy_df = accuracy_df.transpose()
        accuracy_df.columns = [ "weighted_mcc", "weighted_logloss", "weighted_bs", "mcc", "logloss", "brier_score_loss", "tn", "fp", "fn", "tp", "fpr", "fnr", "train_tn", 'train_fp', "train_fn", "train_tp","train_fpr","train_fnr"]
        if test_instance_weight.unique().size > 1:
                total_loss = test_instance_weight[yhat!=y_test].sum()
                fp_loss = test_instance_weight[(yhat!=y_test) & (yhat==1) ].sum()
                fn_loss = test_instance_weight[(yhat!=y_test) & (yhat == 0)].sum()
                accuracy_df["total_loss"] = total_loss
                accuracy_df["fp_loss"] = fp_loss
                accuracy_df["fn_loss"] = fn_loss
        return accuracy_df


def multiclass_accuracy_analysis(classifier, X_test, y_test ):
        pred_proba = classifier.predict(X_test)
        yhat = list(pred_proba.argmax(axis = 1))
        return pd.DataFrame(confusion_matrix(y_test, yhat))


    
def get_feature_importance(classifier):
        feature_importance = pd.DataFrame({'Features': classifier.feature_name(),'Importances': classifier.feature_importances()})
        feature_importance.sort_values(by='Importances', inplace=True,ascending = False )
        return feature_importance


def get_warm_start_parameter():
    # 1 find if the warn start parameter json file exist
    # 2 if exist, read the parameters 
    # 3 if not return empty string e.g. parameters = ""
    parameters = ""
    return parameters

def train_profit_loss_binary(data_path, dictionary_path, hyperparameters, model_dir, new_customer):
    # Load and label data 
    df = pd.read_csv(data_path, low_memory= False)
    # df = df.sample(frac=0.1, random_state=42) #testing for memory usage
    # create components
    warm_starting_param = get_warm_start_parameter()
    random_state = 456   
    fixed_param = {
    'objective': 'binary',
    'metric':  "binary_logloss",
    'boosting_type': 'gbdt',
    'random_state': random_state,
    'verbose': -1,
    'feature_pre_filter': False
    }

    if new_customer:
        columns_to_drop = ["funded_outstanding","priorfundtaphistoryfundedsum","priorfundtaphistorycompletedsum", "priorfundtaphistoryduesum", "priorfundtaphistorypendingsum"]
        df = df.drop(columns = columns_to_drop, errors = 'ignore')
    # full-model 
    X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight,train_holdout,test_holdout = fundtap_train_test_split(df, 
                                                                                                             dictionary_path,
                                                                                                              random_state=456)
    weight = train_instance_weight


    import joblib
    from pathlib import Path
    import json
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 3, 9)  # Further reduced max depth
        max_num_leaves = 2 ** max_depth - 1
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
        param = {
            'objective': 'binary',
            'metric': "binary_logloss",
            'verbose': -1,
            'boosting_type': 'gbdt',
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', 2, min(32, max_num_leaves)),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0,log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0,log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10),
            'seed': random_state
        }
        # param = {
        #     'objective': 'binary',
        #     'metric': "binary_logloss",
        #     'verbosity': -1,
        #     'boosting_type': 'gbdt',
        #     'max_depth': max_depth,
        #     'num_leaves': trial.suggest_int('num_leaves', 2, min(128, max_num_leaves)),
        #     'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        #     'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        #     'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.8),
        #     'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        #     'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        #     'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        #     'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1.0),
        #     'seed': random_state,
        #     'feature_pre_filter': True,  # Let LightGBM decide which features to use
        #     'max_bin': trial.suggest_int('max_bin', 200, 255),
        #     'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 0.5),
        #     'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        #     'max_cat_threshold': trial.suggest_int('max_cat_threshold', 16, 64),  # For categorical features
        #     'cat_l2': trial.suggest_float('cat_l2', 1e-3, 10.0, log=True),  # L2 regularization for categoricals
        #     'cat_smooth': trial.suggest_float('cat_smooth', 1e-3, 10.0, log=True),  # Smoothing for categoricals
        # }

        lgbcv = lgb.cv(param,
                    dtrain,
                    nfold=5,
                    stratified=True,
                    callbacks=[
                        lgb.reset_parameter(learning_rate=learning_rate_decay_power_0995)
                    ]
                    )

        print("CV Results Keys:", lgbcv.keys())  # Debugging: Print the keys of the CV results
    
        score_mean = "binary_logloss-mean"
        score_stdv = "binary_logloss-stdv"
        # check memory usage
        print_memory_usage()
        if score_mean in lgbcv and score_stdv in lgbcv:
            cv_score = lgbcv[score_mean][-1] + lgbcv[score_stdv][-1]
        else:
            print(f"Keys {score_mean} and {score_stdv} not found in CV results.")
            cv_score = float('inf')  # Assign a large value to ensure this trial is not selected

        return cv_score
       
    
    study = optuna.create_study(direction="minimize")  # default TPE sampleR
    # once the warm starting parameter is implemented
    # 1 check if warm_starting_param is empty 
    # if yes use study.enqueue_trial({**fixed_param})
    # if not use study.enqueue_trial({**fixed_param, **warm_starting_param})
    study.enqueue_trial({**fixed_param})
    n_trials = 50
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs = 3)

    best_params = study.best_params

    dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
    classifier = lgb.train({**fixed_param, **best_params}, dtrain)
    accuracy = accuracy_analysis(
        classifier, X_train, X_test, y_train, y_test, test_instance_weight)

    # save components
    final_dtrain = lgb.Dataset(pd.concat([X_train, X_test]), label=pd.concat([y_train, y_test]), weight=pd.concat([train_instance_weight, test_instance_weight]))
    final_clf = lgb.train({**fixed_param, **best_params}, final_dtrain)

    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    if new_customer:
        file_prefix = "new_customer_profitloss"
    else:
        file_prefix = "existing_customer_profitloss"
    joblib.dump(final_clf, Path(model_dir, file_prefix+"classifier.joblib"))
    accuracy.to_csv(Path(model_dir, file_prefix+"accuracy.csv"))
    with open(Path(model_dir, file_prefix+"hyperparamter.json"), 'w') as fp:
        json.dump({**best_params}, fp)

        
    