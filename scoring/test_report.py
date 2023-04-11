import time
import numpy as np
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm
from embeddings.n_way_shot_learning_mean import n_way_shot_learning
from embeddings.n_way_shot_learning_all import n_way_shot_learning_all


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
            predicted.append(n_way_shot_learning_all(net, device, img, dict, threshold, k))
        else:
            predicted.append(n_way_shot_learning(net, device, img, dict, threshold))
        end_time_image = time.time()
        avg_time_image.append(end_time_image-start_time_image)

    end_time = time.time()
    execution_time = end_time - start_time
    unique_classes =  np.unique(actual)

    return classification_report(actual, predicted, labels=unique_classes), execution_time, avg_time_image
