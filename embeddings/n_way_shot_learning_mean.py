import torch
import torch.nn.functional as F

def n_way_shot_learning(net, device, image, embedding_dict, threshold):
    
    output = net.forward_once(image.to(device))
    
    labels = list(embedding_dict.keys())
    embedding_tensors = torch.stack(list(embedding_dict.values()))
    
    scores = F.pairwise_distance(output, embedding_tensors)
    dict_results = dict(zip(labels,scores))

    dict_results_threshold = {k:-v for k,v in dict_results.items() if v < threshold}
    if len(dict_results_threshold) == 0:
        return "Unknown Class"
    values = torch.tensor(list(dict_results_threshold.values()))
    values_softmax = F.softmax(values)
    
    final_dict = {k:v.item() for k,v in zip(dict_results_threshold.keys(),values_softmax)}
    return max(final_dict,key=final_dict.get)