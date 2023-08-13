import torch.nn as nn
from transformers import BertModel
import torch

class JointABSA(nn.Module):
    def __init__(self, num_aspect_labels, num_polarity_labels):
        super(JointABSA, self).__init__()
        
        # Load the BERT model
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Token classification head for aspect term extraction
        self.aspect_classifier = nn.Linear(self.bert.config.hidden_size, num_aspect_labels)
        
        # Token classification head for polarity detection
        self.polarity_classifier = nn.Linear(self.bert.config.hidden_size, num_polarity_labels)

    def forward(self, input_ids, attention_mask):
        # Get BERT's output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        # Token-level predictions for aspect term extraction
        aspect_logits = self.aspect_classifier(sequence_output)
        
        # Token-level predictions for polarity detection
        polarity_logits = self.polarity_classifier(sequence_output)
        
        # Mask polarity logits for non-aspect terms
        aspect_preds = torch.argmax(aspect_logits, dim=-1)
        mask = (aspect_preds == 1).unsqueeze(-1).expand(polarity_logits.shape)
        polarity_logits = polarity_logits * mask
        
        return aspect_logits, polarity_logits
