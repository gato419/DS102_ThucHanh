import numpy as np

def evaluate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0
    f1_macro = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1_macro": f1_macro
    }