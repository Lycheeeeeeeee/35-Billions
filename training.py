"""
TRAINING FUNCTIONS: this file in run in 'script mode' when `.fit` is called
from the notebook. `parse_args` and `train_fn` are called in the
`if __name__ =='__main__'` block.
"""
import argparse
import joblib
from lightgbm import LGBMClassifier
import os
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from training_helper import *
from result_interpretation_helper import *
from hp_tuning_helper import *
from preprocessing_helper import *
import optuna
import json
import gc
from package.data import schemas, datasets
import shap
import matplotlib.pyplot as plt

shap.initjs()


NUMERICAL_TYPES = set(["boolean", "integer", "number"])
CATEGORICAL_TYPES = set(["string"])


class AsTypeFloat32(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype("float32")


def get_numerical_idxs(data_schema):
    idxs = get_idxs(data_schema, NUMERICAL_TYPES)
    return idxs


def get_categorical_idxs(data_schema):
    idxs = get_idxs(data_schema, CATEGORICAL_TYPES)
    return idxs


def get_idxs(data_schema, types):
    idxs = []
    for idx, type in enumerate(data_schema.item_types):
        if type in types:
            idxs.append(idx)
    return idxs


def create_preprocessor(data_schema) -> ColumnTransformer:
    numerical_idxs = get_numerical_idxs(data_schema)
    numerical_transformer = AsTypeFloat32()
    categorical_idxs = get_categorical_idxs(data_schema)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_idxs),
            ("categorical", categorical_transformer, categorical_idxs),
        ],
        remainder="drop",
    )
    return preprocessor


def preprocess_numerical_schema(preprocessor, data_schema):
    num_idx = [e[0] for e in preprocessor.transformers].index("numerical")
    numerical_idxs = get_numerical_idxs(data_schema)
    numerical_items = [data_schema.items[idx] for idx in numerical_idxs]
    features = []
    for item in numerical_items:
        feature = {
            "title": item["title"],
            "description": item["description"],
            "type": "number"
        }
        features.append(feature)
    return num_idx, features


def preprocess_categorical_schema(preprocessor, data_schema):
    cat_idx = [e[0] for e in preprocessor.transformers].index("categorical")
    categorical_idxs = get_categorical_idxs(data_schema)
    categorical_items = [data_schema.items[idx] for idx in categorical_idxs]
    features = []
    ohe = preprocessor.transformers_[cat_idx][1]
    for item, categories in zip(categorical_items, ohe.categories_):
        for category in categories:
            feature = {
                "title": "{}__{}".format(item["title"], category),
                "description": "{} is '{}' if value is 1.0.".format(
                    item["description"].strip('.'), category
                ),
                "type": "number"
            }
            features.append(feature)
    return cat_idx, features


def transform_schema(preprocessor, data_schema):
    num_idx, num_features = preprocess_numerical_schema(preprocessor, data_schema)  # noqa
    cat_idx, cat_features = preprocess_categorical_schema(preprocessor, data_schema)  # noqa
    assert num_idx < cat_idx, "Ordering should be numerical, then categorical."
    features = num_features + cat_features

    array_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "array",
        "minItems": len(features),
        "maxItems": len(features),
        "items": features,
        "title": data_schema.title,
        "description": data_schema.description.replace(
            "items", "features"
        ),
    }
    return schemas.Schema(array_schema)


def load_schemas(schemas_folder):
    data_schema_filepath = Path(schemas_folder, "data.schema.json")
    data_schema = schemas.from_json_schema(data_schema_filepath)
    label_schema_filepath = Path(schemas_folder, "label.schema.json")
    label_schema = schemas.from_json_schema(label_schema_filepath)
    return data_schema, label_schema


def log_cross_val_auc(clf, X, y, cv_splits, log_prefix):
    cv_auc = cross_val_score(clf, X, y, cv=cv_splits, scoring='roc_auc')
    cv_auc_mean = cv_auc.mean()
    cv_auc_error = cv_auc.std() * 2
    log = "{}_auc_cv: {:.5f} (+/- {:.5f})"
    print(log.format(log_prefix, cv_auc_mean, cv_auc_error))


def log_auc(clf, X, y, log_prefix):
    y_pred_proba = clf.predict_proba(X)
    auc = roc_auc_score(y, y_pred_proba[:, 1])
    log = '{}_auc: {:.5f}'
    print(log.format(log_prefix, auc))


def train_pipeline(pipeline, X, y, cv_splits):
    # fit pipeline to cross validation splits
    if cv_splits > 1:
        log_cross_val_auc(pipeline, X, y, cv_splits, 'train')
    # fit pipeline to all training data
    pipeline.fit(X, y)
    log_auc(pipeline, X, y, 'train')
    return pipeline


def test_pipeline(pipeline, X, y):
    log_auc(pipeline, X, y, 'test')


def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tree-boosting-type",
        type=str,
        default="gbdt"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--schemas",
        type=str,
        default=os.environ.get("SM_CHANNEL_SCHEMAS")
    )
    parser.add_argument(
        "--data-train",
        type=str,
        default=os.environ.get("SM_CHANNEL_DATA_TRAIN"),
    )
    parser.add_argument(
        "--hyperparameters",
        type=str,
        default="",
    )
    parser.add_argument(
        "--label-train",
        type=str,
        default=os.environ.get("SM_CHANNEL_LABEL_TRAIN"),
    )
    parser.add_argument(
        "--data-test",
        type=str,
        default=os.environ.get("SM_CHANNEL_DATA_TEST")
    )
    parser.add_argument(
        "--label-test",
        type=str,
        default=os.environ.get("SM_CHANNEL_LABEL_TEST"),
    )

    parser.add_argument(
        "--ramdom-state",
        type=str,
        default=456,
    )

    args, _ = parser.parse_known_args(sys_args)
    return args

def get_shap_values(X_train,X_test,final_clf,model_dir,file_prefix):

    X= X_train.append(X_test)
    explainer = shap.TreeExplainer(final_clf, feature_perturbation = 'interventional')
    shap_values = explainer.shap_values(X)

    vals = np.abs(np.array(shap_values)).mean(1)[0]
    feature_names = X_train.columns

    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)

    shap.summary_plot(shap_values[1], X,max_display=40, show=False)
    plt.savefig(Path(model_dir, file_prefix+".png"), bbox_inches='tight')
    plt.close()
    X.to_csv(Path(model_dir, file_prefix+"X.csv"))
    for name in list(feature_importance.col_name)[0:30]:
        shap.dependence_plot(name, shap_values[1], X, display_features=X, show=False)
        plt.savefig(Path(model_dir, file_prefix+"_" +name+".png"), bbox_inches='tight')
        plt.close()

def train_profit_loss_binary(train_data_path, hyperparameters,model_dir, new_customer):
    ## load data

    df = datasets.read_csv_dataset(train_data_path)

    df = df[df.fundtapprofitloss.notnull() & df.fundtapprofitloss != 0]
    df = preprocessing(df)
    df["label"] = df.fundtapprofitloss >= 0

    # create components
    warm_starting_param = ""
    if hyperparameters != "":
        warm_starting_param = datasets.read_hyper_parameters(hyperparameters)
    else:
        warm_starting_param = {"bagging_fraction": 0.5492535456145099, "bagging_freq": 2, "feature_fraction": 0.88, "lambda_l1": 0.00011548574578690704, "lambda_l2": 1.3199945533897172e-06, "max_depth": 12, "min_child_samples": 10, "num_leaves": 48}
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
    X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight = fundtap_train_test_split(df)
    weight = train_instance_weight
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 14)
        max_num_leaves = (2 ** max_depth) - 1
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
        param = {
            'objective': 'binary',
            'metric': "binary_logloss",
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', 2, max_num_leaves),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'min_sum_hessian_in_leaf': trial.suggest_uniform('min_sum_hessian_in_leaf', 1e-3, 10),
            'seed': random_state
        }

        lgbcv = lgb.cv(param,
                    dtrain,
                    nfold=5,
                    stratified=True,
                    verbose_eval=False,
                    early_stopping_rounds=100,
                    num_boost_round=10000,
                    callbacks=[lgb.reset_parameter(
                        learning_rate=learning_rate_decay_power_0995)]
                    )
        score_mean = "binary_logloss-mean"
        score_stdv = "binary_logloss-stdv"
        cv_score = lgbcv[score_mean][-1] + lgbcv[score_stdv][-1]
        return cv_score
    
    study = optuna.create_study(direction="minimize")  # default TPE sampleR
    study.enqueue_trial({**fixed_param, **warm_starting_param})
    n_trials = 10
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs = 3)

    best_params = study.best_params

    dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
    classifier = lgb.train({**fixed_param, **best_params}, dtrain)
    accuracy = accuracy_analysis(
        classifier, X_train, X_test, y_train, y_test, test_instance_weight)

    # save components
    final_dtrain = lgb.Dataset(X_train.append(X_test), label=y_train.append(y_test), weight=train_instance_weight.append(test_instance_weight))
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

    get_shap_values(X_train,X_test,final_clf,model_dir,file_prefix)


def train_profit_loss_multi(train_data_path, hyperparameters,model_dir, new_customer):
    ## load data
    def encoding_label(x):
        if x <4:
            return x
        else:
            return 4
    
    df = datasets.read_csv_dataset(train_data_path)

    df = df[df.fundtapprofitloss.notnull() & df.fundtapprofitloss != 0]
    df = preprocessing(df, method="multi")
    if "weekspastdue" not in df.columns:
        return 1
    df["label"] = df.weekspastdue
    df['label'] = df['label'].apply(lambda x: encoding_label(x))

    # create components
    warm_starting_param = ""
    if hyperparameters != "":
        warm_starting_param = datasets.read_hyper_parameters(hyperparameters)
    else:
        warm_starting_param = {'max_depth': 14,
                                'num_leaves': 3445,
                                'lambda_l1': 0.0024719526504884707,
                                'lambda_l2': 8.956806049714798e-06,
                                'feature_fraction': 0.15930975035202272,
                                'bagging_fraction': 0.24596777131151706,
                                'bagging_freq': 0,
                                'min_child_samples': 49,
                                'min_sum_hessian_in_leaf': 1.5433498652572106}
                                    
    random_state = 456   
    fixed_param = {
    'objective': 'multiclass',
    'metric':  "multi_logloss",
    'num_classes': 5,
    'boosting_type': 'gbdt',
    'random_state': random_state,
    'verbose': -1,
    'feature_pre_filter': False
    }

    if new_customer:
        columns_to_drop = ["funded_outstanding","priorfundtaphistoryfundedsum","priorfundtaphistorycompletedsum", "priorfundtaphistoryduesum", "priorfundtaphistorypendingsum"]
        df = df.drop(columns = columns_to_drop, errors = 'ignore')
    # full-model 
    X_train, X_test, y_train, y_test, train_instance_weight, test_instance_weight = fundtap_train_test_split(df, hold_out_columns=["quote","fundtapprofitloss","weekspastdue"] , random_state = random_state)
    weight = train_instance_weight
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 2, 14)
        max_num_leaves = (2 ** max_depth) - 1
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
        param = {
            'objective': 'multiclass',
            'metric': "multi_logloss",
            'num_classes': 5,
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', 2, max_num_leaves),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'min_sum_hessian_in_leaf': trial.suggest_uniform('min_sum_hessian_in_leaf', 1e-3, 10),
            'seed': random_state
        }

        lgbcv = lgb.cv(param,
                    dtrain,
                    nfold=3,
                    stratified=True,
                    verbose_eval=False,
                    early_stopping_rounds=100,
                    num_boost_round=10000,
                    callbacks=[lgb.reset_parameter(
                        learning_rate=learning_rate_decay_power_0995)]
                    )
        score_mean = "multi_logloss-mean"
        score_stdv = "multi_logloss-stdv"
        cv_score = lgbcv[score_mean][-1] + lgbcv[score_stdv][-1]
        return cv_score
    
    study = optuna.create_study(direction="minimize")  # default TPE sampleR
    study.enqueue_trial({**fixed_param, **warm_starting_param})
    n_trials = 10
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs = 3)

    best_params = study.best_params

    dtrain = lgb.Dataset(X_train, label=y_train, weight=weight)
    classifier = lgb.train({**fixed_param, **best_params}, dtrain)
    accuracy = multiclass_accuracy_analysis(classifier, X_test, y_test )
    # save components
    final_dtrain = lgb.Dataset(X_train.append(X_test), label=y_train.append(y_test), weight=train_instance_weight.append(test_instance_weight))
    final_clf = lgb.train({**fixed_param, **best_params}, final_dtrain)
    
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    if new_customer:
        file_prefix = "new_customer_overdue"
    else:
        file_prefix = "existing_customer_overdue"
    joblib.dump(final_clf, Path(model_dir, file_prefix+"classifier.joblib"))
    accuracy.to_csv(Path(model_dir, file_prefix+"accuracy.csv"))
    with open(Path(model_dir, file_prefix+"hyperparamter.json"), 'w') as fp:
        json.dump({**best_params}, fp)
    get_shap_values(X_train,X_test,final_clf,model_dir,file_prefix)

def train_fn(args):
    train_profit_loss_binary(args.data_train, args.hyperparameters,args.model_dir, new_customer = True)
    gc.collect()
    train_profit_loss_binary(args.data_train, args.hyperparameters,args.model_dir, new_customer = False)
    gc.collect()
    train_profit_loss_multi(args.data_train, args.hyperparameters,args.model_dir, new_customer = True)
    gc.collect()
    train_profit_loss_multi(args.data_train, args.hyperparameters,args.model_dir, new_customer = False)
    gc.collect()