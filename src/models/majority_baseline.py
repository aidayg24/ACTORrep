from transformers import AutoModelForSequenceClassification


def build_majority_model(model_name="bert-base-uncased", num_labels=2):
    """
    Build a standard BERT sequence classification model
    for the majority-vote baseline.

    :param model_name
    :param num_labels
    :return: model
    """
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    return model
