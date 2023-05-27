def extract_labels_and_embeddings(input_dict):
    labels = []
    embeddings = []
    for key, values in input_dict.items():
        for value in values:
            labels.append(value)
            embeddings.append(key)
    return labels, embeddings