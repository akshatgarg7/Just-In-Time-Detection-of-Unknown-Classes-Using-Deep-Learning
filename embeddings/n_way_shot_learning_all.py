import torch
import torch.nn.functional as F
import heapq
from collections import Counter
from embeddings.extract_labels_embeddings import extract_labels_and_embeddings

def n_way_shot_learning_all(net, device, image, embedding_dict, threshold, k):

    embeddings, labels = extract_labels_and_embeddings(embedding_dict)

    output = net.forward_once(image.to(device))
    
    embedding_tensors = torch.stack(embeddings)
    scores = F.pairwise_distance(output, embedding_tensors)
    list_tuples = list(zip(labels,scores))
    top_k = heapq.nsmallest(k, list_tuples, key=lambda tup:tup[1])
    list_tuples_after_filter = [(tup[0],-tup[1]) for tup in top_k if tup[1]<threshold]

    if not list_tuples_after_filter:
        return "Unknown Class"
    
    # Extract the labels from the list of tuples after filters
    labels_after_top = [t[0] for t in list_tuples_after_filter]

    # Count the occurrences of each label and find the most common one
    label_counts = Counter(labels_after_top)

    # Find the highest count (mode count)
    max_count = label_counts.most_common(1)[0][1]

    # Get all modes by filtering labels with the highest count
    modes = [label for label, count in label_counts.items() if count == max_count]
    if len(modes)==1:
        return modes[0]
    
    else:
        # Group the tuples by their label
        label_groups = defaultdict(list)
        for t in list_tuples_after_filter:
            label_groups[t[0]].append(t)
        
        mode_min_values = [(mode, max([t[1] for t in label_groups[mode]])) for mode in modes]
        # Convert the list of tuples into a dictionary
        result_dict = {t[0]: t[1] for t in mode_min_values}
        values = torch.tensor(list(result_dict.values()))
        softmax_vals = F.softmax(values)
        final_dict = {k:v.item() for k,v in zip(result_dict.keys(),softmax_vals)}
        return max(final_dict,key=final_dict.get)