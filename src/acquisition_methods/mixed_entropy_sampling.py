"""
ACTOR-style mixed entropy acquisition.

This method combines:
1. individual-level entropy: uncertainty of the specific annotator head
2. group-level entropy: uncertainty of the aggregated group prediction

Following Wang & Plank:
H_mix = H_indi + H_group

For each unlabeled (text, annotator) row, we compute:
score = individual_entropy_for_that_annotator + group_entropy_for_all_heads

Then we select rows with the highest mixed score.
"""

import torch
import torch.nn.functional as F


def mixed_entropy_sampling(
    model,
    dataloader,
    count=100,
    device=None,
):
    """
    Select top-k unlabeled rows using mixed individual + group entropy.

    Expected dataloader batch:
        input_ids, attention_mask, original_indices, annotator_idx

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
            annotator_idx = annotator_idx.to(device)

            batch_size = input_ids.size(0)

            # -----------------------------
            # 1. Individual-level entropy
            # -----------------------------
            individual_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                annotator_idx=annotator_idx,
            )

            if hasattr(individual_outputs, "logits"):
                individual_logits = individual_outputs.logits
            elif isinstance(individual_outputs, dict) and "logits" in individual_outputs:
                individual_logits = individual_outputs["logits"]
            elif isinstance(individual_outputs, tuple):
                individual_logits = individual_outputs[0]
            else:
                individual_logits = individual_outputs

            individual_probs = F.softmax(individual_logits, dim=-1)

            individual_entropy = -(
                individual_probs * torch.log(individual_probs + 1e-12)
            ).sum(dim=-1)

            # -----------------------------
            # 2. Group-level entropy
            # -----------------------------
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

            # Paper-faithful group entropy:
            # normalize logits, sum them, softmax, entropy
            normalized_logits = F.normalize(
                all_head_logits,
                p=2,
                dim=-1,
            )

            group_logits = normalized_logits.sum(dim=1)

            group_probs = F.softmax(group_logits, dim=-1)

            group_entropy = -(
                group_probs * torch.log(group_probs + 1e-12)
            ).sum(dim=-1)

            # -----------------------------
            # 3. Mixed entropy
            # -----------------------------
            mixed_entropy = individual_entropy + group_entropy

            all_scores.append(mixed_entropy.detach().cpu())
            all_indices.extend(original_indices.tolist())

    all_scores = torch.cat(all_scores)

    k = min(count, len(all_scores))
    top_positions = torch.topk(all_scores, k=k).indices.tolist()

    selected_indices = [all_indices[pos] for pos in top_positions]

    return selected_indices