"""
group-level entropy acquisition.

This method is for the multi-head annotator-specific model.

Instead of measuring uncertainty from only one annotator head, we combine
the predictions of all annotator heads for each unlabeled example and then
compute entropy over the aggregated group prediction.

Wang & Plank's ACTOR paper
"""

import torch
import torch.nn.functional as F


def group_entropy_sampling(
    model,
    dataloader,
    count=100,
    device=None,
):
    """
    Select top-k unlabeled rows using ACTOR group-level entropy.

    Expected dataloader batch:
        input_ids, attention_mask, original_indices, annotator_idx

    Note:
        Even though the dataloader includes annotator_idx, group entropy
        does not use only that annotator's head. Instead, it evaluates
        all annotator heads for the same input and aggregates them.

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

            logits_per_head = []

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

                logits_per_head.append(logits)

            # Shape: [batch_size, num_annotators, num_labels]
            all_head_logits = torch.stack(logits_per_head, dim=1)

            # 1. Normalize each annotator head's logits
            # z_norm^h
            normalized_logits = torch.nn.functional.normalize(
                all_head_logits,
                p=2,
                dim=-1
            )

            # 2. Sum normalized logits across annotator heads
            # z_group = sum_h z_norm^h
            group_logits = normalized_logits.sum(dim=1)

            # 3. Convert group logits to probabilities
            # p_i(x) = softmax(z_i(x))
            group_probs = torch.nn.functional.softmax(group_logits, dim=-1)

            # 4. Compute group-level entropy
            # H_group(x) = - sum_i p_i(x) log p_i(x)

            group_entropy = -(
                    group_probs * torch.log(group_probs + 1e-12)
            ).sum(dim=-1)

            all_scores.append(group_entropy.detach().cpu())
            all_indices.extend(original_indices.tolist())

    all_scores = torch.cat(all_scores)

    k = min(count, len(all_scores))
    top_positions = torch.topk(all_scores, k=k).indices.tolist()

    selected_indices = [all_indices[pos] for pos in top_positions]

    return selected_indices