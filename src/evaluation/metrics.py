import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    """
     Compute evaluation metrics.
    :param eval_pred:
    :return:
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }
