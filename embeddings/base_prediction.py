import torch
import torch.nn.functional as F
import heapq
from collections import Counter
from embeddings.extract_labels_embeddings import extract_labels_and_embeddings

def base_prediction(net, device, image, embedding_dict, threshold):

    embeddings, labels = extract_labels_and_embeddings(embedding_dict)

    output = net.forward_once(image.to(device))
    
    embedding_tensors = torch.stack(embeddings)
    scores = F.pairwise_distance(output, embedding_tensors)
    list_tuples = list(zip(labels,scores))
    list_tuples_after_filter = [(tup[0],-tup[1]) for tup in list_tuples if tup[1]<threshold]

    if not list_tuples_after_filter:
        return "Unknown Class"
    
    else:
        max_tuple = max(list_tuples_after_filter, key=lambda x: x[1])
        return max_tuple[0]