import torch
import torch.nn.functional as F
import numpy as np

# Defining Contrastive loss with Cosine Similarity 
class ContrastiveLossCosine(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        cosine_sim = torch.cosine_similarity(output1, output2)
        cosine_sim = 1 - cosine_sim
        loss = torch.mean((1 - label) * torch.pow(cosine_sim, 2) + 
                          label * torch.pow(torch.clamp(self.margin - cosine_sim, min=0.0), 2))
        return loss