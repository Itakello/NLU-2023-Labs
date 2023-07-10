import torch.nn as nn
from transformers import BertModel

class BertForIntentAndSlotFilling(nn.Module):
    def __init__(self, out_slot, out_int, dropout_rate=0.1):
        super(BertForIntentAndSlotFilling, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_rate)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, out_int)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # a sequence of hidden-states at the output of the last layer of the model
        # ! slot filling task
        sequence_output = self.dropout(outputs[0])
        # a pooled output of the model (last layer hidden-state of the first token of the sequence, classification token)
        # ! intent classification task
        pooled_output = self.dropout(outputs[1])

        slot_logits = self.slot_classifier(sequence_output) # (batch_size, sequence_length, num_slot_classes)
        intent_logits = self.intent_classifier(pooled_output) # (batch_size, num_intent_classes)

        return slot_logits, intent_logits
