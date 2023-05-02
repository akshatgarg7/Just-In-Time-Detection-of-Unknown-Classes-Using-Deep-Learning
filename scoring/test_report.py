import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.notebook import tqdm
from embeddings.n_way_shot_learning_mean import n_way_shot_learning
from embeddings.top_k_selection import top_k_selection
from embeddings.base_prediction import base_prediction


from utils.other_utils import joinpath, get_class_names

def test_report(PATH, net, device, dataloader, dataloader_size, dict, threshold, embed_flag, k=10):
    known_class = get_class_names(joinpath(PATH,'train'))
    start_time = time.time()

    actual = []
    predicted = []
    avg_time_image = []

    dataiter = iter(dataloader)
    for i in tqdm(range(len(dataloader_size))):
        start_time_image = time.time()
        img, label = next(dataiter)
        
        if label[0] in known_class:
            actual.append(label[0])
        else:
            actual.append('Unknown Class')
        
        if embed_flag == 'all':
            predicted.append(top_k_selection(net, device, img, dict, threshold, k))
        elif embed_flag == 'base':
            predicted.append(base_prediction(net, device, img, dict, threshold))
        else:
            predicted.append(n_way_shot_learning(net, device, img, dict, threshold))
        end_time_image = time.time()
        avg_time_image.append(end_time_image-start_time_image)
    
    end_time = time.time()
    execution_time = end_time - start_time
    unique_classes =  np.unique(actual)

    cm = confusion_matrix(actual, predicted, labels=unique_classes)
    cm = np.vstack((unique_classes, cm))

    return classification_report(actual, predicted, labels=unique_classes), execution_time, avg_time_image, cm
