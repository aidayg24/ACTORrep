import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer


def build_unlabeled_loader(
    df,
    text_column="text",
    id_column="original_index",
    annotator_column="annotator_idx",
    tokenizer_name="bert-base-uncased",
    max_length=128,
    batch_size=32,
    include_annotator=True,
):
    """
    Build unlabeled acquisition dataloader:
    returns batches of (inputs, comment_ids)

    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    input_ids_list = []
    attention_mask_list = []
    comment_ids_list = []
    annotator_idx_list = []

    for _, row in df.iterrows():
        enc = tokenizer(
            row[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids_list.append(enc["input_ids"].squeeze(0))
        attention_mask_list.append(enc["attention_mask"].squeeze(0))
        comment_ids_list.append(int(row[id_column]))

        if include_annotator:
            annotator_idx_list.append(int(row[annotator_column]))

    input_ids_tensor = torch.stack(input_ids_list)
    attention_mask_tensor = torch.stack(attention_mask_list)
    comment_ids_tensor = torch.tensor(comment_ids_list, dtype=torch.long)

    if include_annotator:
        annotator_idx_tensor = torch.tensor(annotator_idx_list, dtype=torch.long)
        dataset = TensorDataset(
            input_ids_tensor,
            attention_mask_tensor,
            comment_ids_tensor,
            annotator_idx_tensor,
        )
    else:
        dataset = TensorDataset(
            input_ids_tensor,
            attention_mask_tensor,
            comment_ids_tensor,
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)