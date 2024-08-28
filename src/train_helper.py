from memory_helper import print_memory_usage
import hp_tuning_helper
import data_helper
import result_interpretation_helper
import joblib
from pathlib import Path
import json
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np



def train_profit_loss_binary_on_weighted_mcc(data_path, dictionary_path, model_dir, new_customer, n_trials = 10):
    # Load and label data 
    df = pd.read_csv(data_path, low_memory= False)

    random_state = 456   

    # full-model 
    X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight,train_holdout,test_holdout = data_helper.fundtap_train_test_split(df, 
                                                                                                             dictionary_path,
                                                                                                              random_state=456)
    weight = train_instance_weight

    def learning_rate_decay_power_0995(current_iter): 
        base_learning_rate = 0.1
        lr = base_learning_rate * np.power(.995, current_iter) 
        return lr if lr > 1e-3 else 1e-3
    from sklearn.metrics import matthews_corrcoef

    def weighted_mcc(preds, train_data):
        labels = train_data.get_label()
        weights = train_data.get_weight()
        preds_binary = (preds > 0.5).astype(int)
        mcc = matthews_corrcoef(labels, preds_binary, sample_weight=weights)
        return 'weighted_mcc', mcc, True  # True

    fixed_param = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'random_state': random_state,
        'feature_pre_filter': False,
        'max_depth': -1,
        'min_gain_to_split':0.01
        }
    def objective(trial):
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
        param = {
            'objective': 'binary',
            'metric': 'None',  # We'll use our custom metric
            'verbose': -1,
            'feature_pre_filter': False,
            'max_depth': -1,
            'min_gain_to_split': 0.01,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 32, 96),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10),
            'seed': random_state,
        }

        lgbcv = lgb.cv(
            param,
            dtrain,
            feval=weighted_mcc,  # Use our custom evaluation function
            nfold=5,
            stratified=True,
            callbacks=[
                lgb.reset_parameter(learning_rate=learning_rate_decay_power_0995)
            ]
        )

        print("CV Results Keys:", lgbcv.keys())  # Debugging: Print the keys of the CV results
        
        score_mean = "valid weighted_mcc-mean"
        score_stdv = "valid weighted_mcc-stdv"

        # Check memory usage
        print_memory_usage()

        print('____________________________________')
        print(lgbcv[score_mean])
        print('____________________________________')
        print(lgbcv[score_stdv])
        if score_mean in lgbcv and score_stdv in lgbcv:
            cv_score = lgbcv[score_mean][-1] + lgbcv[score_stdv][-1]  # Negative because we want to maximize MCC
        else:
            print(f"Keys {score_mean} and {score_stdv} not found in CV results.")
            cv_score = float('inf')  # Assign a large value to ensure this trial is not selected

        return cv_score
    
    study = optuna.create_study(direction="maximize")  # default TPE sampleR
  
    warm_starting_param = hp_tuning_helper.get_warm_start_parameter(model_dir, new_customer)
    # print(warm_starting_param)
    if warm_starting_param != "":
        study.enqueue_trial({**fixed_param, **warm_starting_param})
        
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs = 1) # set to nunber of cores when testing

    best_params = study.best_params

    dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
    classifier = lgb.train({**fixed_param, **best_params}, dtrain, feval= weighted_mcc)
    accuracy = result_interpretation_helper.accuracy_analysis(classifier, X_train, X_test, y_train, y_test, test_instance_weight)

    # save components
    final_dtrain = lgb.Dataset(pd.concat([X_train, X_test]), label=pd.concat([y_train, y_test]), weight=pd.concat([train_instance_weight, test_instance_weight]))
    final_clf = lgb.train({**fixed_param, **best_params}, final_dtrain, feval= weighted_mcc)

    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    if new_customer:
        file_prefix = "new_customer_profitloss"
    else:
        file_prefix = "existing_customer_profitloss"
    joblib.dump(final_clf, Path(model_dir, file_prefix+"_classifier.joblib"))
    accuracy.to_csv(Path(model_dir, file_prefix+"_accuracy.csv"))
    with open(Path(model_dir, file_prefix+"_hyperparameter.json"), 'w') as fp:
        json.dump({**best_params}, fp)

        
    
def train_profit_loss_binary_on_log_loss(data_path, dictionary_path, model_dir, new_customer, n_trials = 10):
    # Load and label data 
    df = pd.read_csv(data_path, low_memory= False)
    # df = df.sample(frac=0.1, random_state=42) #testing for memory usage
    # create components

    random_state = 456   

    # full-model 
    X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight,train_holdout,test_holdout = data_helper.fundtap_train_test_split(df, 
                                                                                                             dictionary_path,
                                                                                                              random_state=456)
    weight = train_instance_weight

    def learning_rate_decay_power_0995(current_iter): 
        base_learning_rate = 0.1
        lr = base_learning_rate * np.power(.995, current_iter) 
        return lr if lr > 1e-3 else 1e-3
    fixed_param = {
        'objective': 'binary',
        'metric':  "binary_logloss",
        'boosting_type': 'gbdt',
        'random_state': random_state,
        'verbose': -1,
        'feature_pre_filter': False,
        'max_depth': -1,
        'min_gain_to_split':0.01
    }
    def objective(trial):
        # max_depth = trial.suggest_int('max_depth', 3, 9)  # Further reduced max depth
        # max_num_leaves = 2 ** max_depth - 1
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
        param = {
            'objective': 'binary',
            'metric': "binary_logloss",
            'verbose': -1,
            'feature_pre_filter': False,
            'max_depth': -1,
            'min_gain_to_split':0.01,
            # 'max_depth': max_depth,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 2, 64),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0,log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0,log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10),
            'seed': random_state,
        }

        lgbcv = lgb.cv(param,
                    dtrain,
                    nfold=10,
                    stratified=True,
                    callbacks=[
                        lgb.reset_parameter(learning_rate=learning_rate_decay_power_0995)
                    ]
                    )

        print("CV Results Keys:", lgbcv.keys())  # Debugging: Print the keys of the CV results
    
        score_mean = "valid binary_logloss-mean"
        score_stdv = "valid binary_logloss-stdv"

        # check memory usage
        print_memory_usage()
        # print('lgbcv', lgbcv)
        if score_mean in lgbcv and score_stdv in lgbcv:
            cv_score = lgbcv[score_mean][-1] + lgbcv[score_stdv][-1]
        else:
            print(f"Keys {score_mean} and {score_stdv} not found in CV results.")
            cv_score = float('inf')  # Assign a large value to ensure this trial is not selected

        return cv_score
       
    
    study = optuna.create_study(direction="minimize")  # default TPE sampleR
  
    warm_starting_param = hp_tuning_helper.get_warm_start_parameter(model_dir, new_customer)
    # print(warm_starting_param)
    if warm_starting_param != "":
        study.enqueue_trial({**fixed_param, **warm_starting_param})
        
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs = 1) # set to nunber of cores when testing

    best_params = study.best_params

    dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
    classifier = lgb.train({**fixed_param, **best_params}, dtrain)
    accuracy = result_interpretation_helper.accuracy_analysis(classifier, X_train, X_test, y_train, y_test, test_instance_weight)

    # save components
    final_dtrain = lgb.Dataset(pd.concat([X_train, X_test]), label=pd.concat([y_train, y_test]), weight=pd.concat([train_instance_weight, test_instance_weight]))
    final_clf = lgb.train({**fixed_param, **best_params}, final_dtrain)

    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    if new_customer:
        file_prefix = "new_customer_profitloss"
    else:
        file_prefix = "existing_customer_profitloss"
    joblib.dump(final_clf, Path(model_dir, file_prefix+"_classifier.joblib"))
    accuracy.to_csv(Path(model_dir, file_prefix+"_accuracy.csv"))
    with open(Path(model_dir, file_prefix+"_hyperparameter.json"), 'w') as fp:
        json.dump({**best_params}, fp)

        
    