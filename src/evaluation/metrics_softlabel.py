import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def compute_metrics_softlabel(eval_pred):
    """
    Compute evaluation metrics for soft-label training.
    """
    logits, labels = eval_pred
    gold_labels = np.argmax(labels, axis=-1)
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(gold_labels, predictions)
    macro_f1 = f1_score(gold_labels, predictions, average="macro")
    weighted_f1 = f1_score(gold_labels, predictions, average="weighted")

    # Soft cross-entropy
    probs = softmax(logits, axis=-1)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    soft_ce = -np.sum(labels * np.log(probs), axis=-1).mean()

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "soft_cross_entropy": soft_ce,
    }
