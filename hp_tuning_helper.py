
import lightgbm as lgb
import numpy as np
import optuna
import optuna.integration.lightgbm as opt_lgb


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
