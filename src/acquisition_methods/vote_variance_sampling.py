"""
ACTOR-style vote variance acquisition.

This method measures disagreement between annotator-specific heads.


This follows Wang & Plank's vote variance method:
Var = (1/H) * sum_h (y^h - mu)^2
where y^h is the prediction of head h.
"""

import torch


def vote_variance_sampling(
    model,
    dataloader,
    count=100,
    device=None,
):
    """
    Select top-k unlabeled rows using vote variance across annotator heads.

    Expected dataloader batch:
        input_ids, attention_mask, original_indices, annotator_idx

    Args:
        model: trained multi-head model
        dataloader: unlabeled dataloader
        count: number of rows to acquire
        device: torch device

    Returns:
        selected_indices: list of original dataframe indices
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_scores = []
    all_indices = []

    num_annotators = len(model.classifier_heads)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, original_indices, annotator_idx = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            batch_size = input_ids.size(0)
            predictions_per_head = []

            for head_id in range(num_annotators):
                head_ids = torch.full(
                    size=(batch_size,),
                    fill_value=head_id,
                    dtype=torch.long,
                    device=device,
                )

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    annotator_idx=head_ids,
                )

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                predictions = torch.argmax(logits, dim=-1)
                predictions_per_head.append(predictions)

            # Shape: [batch_size, num_annotators]
            all_head_predictions = torch.stack(predictions_per_head, dim=1)

            # Convert class predictions to float so variance can be computed
            all_head_predictions = all_head_predictions.float()

            # mu = average prediction across heads
            mean_prediction = all_head_predictions.mean(dim=1, keepdim=True)

            # Var = 1/H * sum_h (y^h - mu)^2
            vote_variance = (
                (all_head_predictions - mean_prediction) ** 2
            ).mean(dim=1)

            all_scores.append(vote_variance.detach().cpu())
            all_indices.extend(original_indices.tolist())

    all_scores = torch.cat(all_scores)

    k = min(count, len(all_scores))
    top_positions = torch.topk(all_scores, k=k).indices.tolist()

    selected_indices = [all_indices[pos] for pos in top_positions]

    return selected_indices