import torch
import torch.nn.functional as F


def entropy_sampling(model, dataloader, device, count=100, model_type="softmax"):
    """
        Select top-k most uncertain unlabeled examples using entropy.
        Assumes model outputs logits of shape [batch_size, num_classes].
        Args:
            model: PyTorch model
            dataloader: PyTorch dataloader
            device: PyTorch device
            count: Number of examples to select
            model_type: Type of model to use

        Returns:
            selected_ids: List of selected examples

        """

    model.eval()
    all_scores = []
    all_ids = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, comment_ids = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # handle common output formats
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs

            if model_type == "softmax":
                probs = F.softmax(logits, dim=1)
            elif model_type == "dirichlet":
                alpha = logits + 1.0
                S = alpha.sum(dim=1, keepdim=True)
                probs = alpha / S
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            ent = -(probs * torch.log(probs + 1e-12)).sum(dim=1)

            all_scores.append(ent.cpu())
            all_ids.extend(comment_ids.tolist())

    all_scores = torch.cat(all_scores)

    top_idx = torch.topk(all_scores, k=min(count, len(all_scores))).indices
    selected_ids = [all_ids[i] for i in top_idx.tolist()]

    return selected_ids
