
# result_interpretation_helper #
from sklearn.metrics import log_loss, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, brier_score_loss
import pandas as pd
import numpy as np

def accuracy_analysis(classifier, X_train, X_test, y_train, y_test, test_instance_weight):
    pred_proba = classifier.predict(X_test)
    yhat = np.where(pred_proba < 0.5, 0, 1)

    mcc = matthews_corrcoef(y_test, yhat)
    logloss = log_loss(y_test, pred_proba)
    bs = brier_score_loss(y_test, pred_proba)

    weighted_mcc = matthews_corrcoef(y_test, yhat, sample_weight=test_instance_weight)
    weighted_logloss = log_loss(y_test, pred_proba, sample_weight=test_instance_weight)
    weighted_bs = brier_score_loss(y_test, pred_proba, sample_weight=test_instance_weight)

    tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
    fpr = fp / float(fp + tn)
    fnr = fn / float(fn + tp)

    train_yhat = np.where(classifier.predict(X_train) < 0.5, 0, 1)
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, train_yhat).ravel()
    train_fpr = train_fp / float(train_fp + train_tn)
    train_fnr = train_fn / float(train_fn + train_tp)

    accuracy_df = pd.DataFrame([[weighted_mcc, weighted_logloss, weighted_bs, mcc, logloss, bs, 
                                 tn, fp, fn, tp, fpr, fnr, train_tn, train_fp, train_fn, train_tp, train_fpr, train_fnr]])
    accuracy_df.columns = ["weighted_mcc", "weighted_logloss", "weighted_bs", "mcc", "logloss", "brier_score_loss", 
                           "tn", "fp", "fn", "tp", "fpr", "fnr", "train_tn", 'train_fp', "train_fn", "train_tp", "train_fpr", "train_fnr"]

   
    total_loss = test_instance_weight[yhat != y_test].sum()
    fp_loss = test_instance_weight[(yhat != y_test) & (yhat == 1)].sum()
    fn_loss = test_instance_weight[(yhat != y_test) & (yhat == 0)].sum()

    accuracy_df["total_loss"] = total_loss
    accuracy_df["fp_loss"] = fp_loss
    accuracy_df["fn_loss"] = fn_loss

    # Print messages about model performance
    print("\nModel Performance Analysis:")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")

    print(f"Weighted_mcc: {weighted_mcc:.4f}")
    
    
    print(f"Total Money Lost: ${total_loss:.2f}")

    fp_loss = test_instance_weight[(yhat != y_test) & (yhat == 1)].sum()
    fn_loss = test_instance_weight[(yhat != y_test) & (yhat == 0)].sum()
    print(f"SUM OF WEIGHT: Money funded False Positives (unprofitable quote predicted as profitable): ${fp_loss:.2f}")

    print(f"SUM OF WEIGHT: Money did not fund due to False Negatives (profitable quote predicted as non-profitable): ${fn_loss:.2f}")

    # Check for overfitting
    train_accuracy = (train_tn + train_tp) / (train_tn + train_fp + train_fn + train_tp)
    test_accuracy = (tn + tp) / (tn + fp + fn + tp)
    
    print("\nOverfitting Analysis:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    if train_accuracy - test_accuracy > 0.05:
        print("Warning: The model may be overfitting. The train accuracy is significantly higher than the test accuracy.")
    elif train_accuracy - test_accuracy > 0.02:
        print("Note: There might be slight overfitting. Keep an eye on the model's performance.")
    else:
        print("Good news: The model doesn't seem to be overfitting.")

    return accuracy_df


def multiclass_accuracy_analysis(classifier, X_test, y_test ):
        pred_proba = classifier.predict(X_test)
        yhat = list(pred_proba.argmax(axis = 1))
        return pd.DataFrame(confusion_matrix(y_test, yhat))


    
def get_feature_importance(classifier):
        feature_importance = pd.DataFrame({'Features': classifier.feature_name(),'Importances': classifier.feature_importances()})
        feature_importance.sort_values(by='Importances', inplace=True,ascending = False )
        return feature_importance

