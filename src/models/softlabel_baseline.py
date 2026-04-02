import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class SoftLabelClassifier(nn.Module):
    """
    soft label model
    BERT-based classifier trained on soft-label distributions.
    """

    def __init__(self, model_name="bert-base-uncased",
                 num_labels=2,
                 dropout_prob=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids
                               )

        # get the cls representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            # labels expected shape: [batch_size, num_labels]
            # logits shape: [batch_size, num_labels]
            log_probs = F.log_softmax(logits, dim=-1)
            # soft cross-entropy:
            # loss = - sum_i target_i * log(pred_i)
            loss = -(labels * log_probs).sum(dim=-1).mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


def build_softlabel_model(model_name="bert-base-uncased", num_labels=2):
    """
    Build a BERT classifier for soft-label training.
    """
    return SoftLabelClassifier(model_name=model_name, num_labels=num_labels)
