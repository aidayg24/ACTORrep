import torch
import torch.nn.functional as F


def bandit_ucb_sampling(
    model,
    dataloader,
    count=100,
    device=None,
    alpha=1.0,
):
    """
    Minimal Linear-UCB-inspired bandit acquisition.

    Version 1 is intentionally simple:
    - builds uncertainty-based features
    - uses a fixed UCB-like score
    - returns selected original dataframe indices

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

            # ----- individual annotator prediction -----
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                annotator_idx=annotator_idx,
            )

            if hasattr(outputs, "logits"):
                individual_logits = outputs.logits
            elif isinstance(outputs, dict) and "logits" in outputs:
                individual_logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                individual_logits = outputs[0]
            else:
                individual_logits = outputs

            individual_probs = F.softmax(individual_logits, dim=-1)

            individual_entropy = -(
                individual_probs * torch.log(individual_probs + 1e-12)
            ).sum(dim=-1)

            max_confidence = individual_probs.max(dim=-1).values

            # ----- group prediction across all annotator heads -----
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

            all_head_logits = torch.stack(logits_per_head, dim=1)

            normalized_logits = F.normalize(all_head_logits, p=2, dim=-1)
            group_logits = normalized_logits.sum(dim=1)
            group_probs = F.softmax(group_logits, dim=-1)

            group_entropy = -(
                group_probs * torch.log(group_probs + 1e-12)
            ).sum(dim=-1)

            predictions_per_head = torch.argmax(all_head_logits, dim=-1).float()
            mean_prediction = predictions_per_head.mean(dim=1, keepdim=True)

            vote_variance = (
                (predictions_per_head - mean_prediction) ** 2
            ).mean(dim=1)

            # ----- simple UCB-style score -----
            #
            # For now:
            # exploitation = group entropy + individual entropy + vote variance
            # exploration = prefer low-confidence examples
            #
            # This is not yet a fully learned bandit.
            # It is the bridge version before adding updateable Linear UCB.
            score = (
                group_entropy
                + individual_entropy
                + vote_variance
                + alpha * (1.0 - max_confidence)
            )

            all_scores.append(score.detach().cpu())
            all_indices.extend(original_indices.tolist())

    all_scores = torch.cat(all_scores)

    k = min(count, len(all_scores))
    top_positions = torch.topk(all_scores, k=k).indices.tolist()

    selected_indices = [all_indices[pos] for pos in top_positions]

    return selected_indices