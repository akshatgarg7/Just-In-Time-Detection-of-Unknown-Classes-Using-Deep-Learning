import torch

def mean_embeddings(dict_embeddings):
    mean_dict = {}

    for key in dict_embeddings:
        mean_dict[key] = torch.mean(torch.stack(dict_embeddings[key]), dim=0)
    
    return mean_dict