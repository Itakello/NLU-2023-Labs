import torch.nn as nn
from transformers import BertModel
import torch

class JointABSA(nn.Module):
    def __init__(self, num_aspect_labels, num_polarity_labels):
        super(JointABSA, self).__init__()
        
        # BERT model for feature extraction
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Aspect head
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, num_aspect_labels)
        
        # Polarity head
        self.polarity_classifier = nn.Linear(self.bert.config.hidden_size, num_polarity_labels)

    def forward(self, input_ids, attention_mask):
        # Passing input through BERT model
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # Aspect classification
        aspect_logits = self.aspect_classifier(outputs.last_hidden_state)
        
        # Polarity classification
        polarity_logits = self.polarity_classifier(outputs.last_hidden_state)

        return aspect_logits, polarity_logits
