import torch
from torch import nn
from transformers import AutoModel


class MultiTaskModel(nn.Module):
    """
    Multi-task model with
    one shared transformer encoder
    one classifier head per annotator
    """

    def __init__(self, model_name="bert-base-uncased",
                 num_labels=2, num_annotators=6,
                 dropout_prob=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier_heads = nn.ModuleList([nn.Linear(hidden_size, num_labels) for _ in range(num_annotators)])
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, annotator_idx, labels=None):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)

        # get the cls representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # route each data to the correct annotator head
        batch_logits = []

        for i in range(pooled_output.size(0)):
            head_idx = annotator_idx[i].item()
            example_repr = pooled_output[i].unsqueeze(0)
            logits = self.classifier_heads[head_idx](example_repr)
            batch_logits.append(logits)

        # combine the logits into one tensor
        logits = torch.cat(batch_logits, dim=0)

        output = {"logits": logits}

        # compute loss
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return output


def build_multitask_model(model_name="bert-base-uncased", num_labels=2, num_annotators=6, dropout_prob=0.1):
    """

    :param model_name:
    :param num_labels:
    :param num_annotators:
    :param dropout_prob:
    :return:
    """
    return MultiTaskModel(
        model_name=model_name,
        num_labels=num_labels,
        num_annotators=num_annotators,
        dropout_prob=dropout_prob
    )
