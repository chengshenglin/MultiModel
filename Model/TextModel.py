import torch.nn as nn
from transformers import XLMRobertaModel

class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        
        self.fc = nn.Sequential(
            nn.Linear(768, args['hidden_size']),
            nn.ReLU(inplace=True)
        )
        
        for param in self.roberta.parameters():
            param.requires_grad = True

    def forward(self, text_input):
        roberta_out = self.roberta(**text_input)
        pooler_output = roberta_out['pooler_output']
        return self.fc(pooler_output)