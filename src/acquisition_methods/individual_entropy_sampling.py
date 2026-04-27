"""
ACTOR- individual-level entropy acquisition.

This acquisition method is for the multi-head / annotator-specific model.
Each unlabeled row corresponds to one (text, annotator_id) pair.

1. calculate the specific annotator uncertainty
z^a = [z_1 ^a , .... z_n ^a]
H_indi =(P^a | x)
argmax H_indi

This implements Wang & Plank's individual-level entropy idea

"""

import torch
import torch.nn.functional as F


def individual_entropy_sampling(
    model,
    dataloader,
    count=100,
    device=None,
):
    """
    Select top-k unlabeled (text, annotator) rows using individual-level entropy.

    Expected dataloader batch:
        input_ids, attention_mask, original_indices, annotator_idx

    Args:
        model: trained multi-head ACTOR-style model
        dataloader: unlabeled dataloader with annotator_idx
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

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, original_indices, annotator_idx = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            annotator_idx = annotator_idx.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                annotator_idx=annotator_idx,
            )

            # implementing their formula
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            probs = F.softmax(logits, dim=-1)

            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

            all_scores.append(entropy.detach().cpu())
            all_indices.extend(original_indices.tolist())

    all_scores = torch.cat(all_scores)

    k = min(count, len(all_scores))
    top_positions = torch.topk(all_scores, k=k).indices.tolist()

    selected_indices = [all_indices[pos] for pos in top_positions]

    return selected_indices