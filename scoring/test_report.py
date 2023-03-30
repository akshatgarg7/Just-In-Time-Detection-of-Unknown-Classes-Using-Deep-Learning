import time
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm
from embeddings.n_way_shot_learning import n_way_shot_learning

def test_report(net, device, dataloader, dataloader_size, dict, threshold):
    start_time = time.time()

    actual = []
    predicted = []
    avg_time_image = []

    dataiter = iter(dataloader)
    for i in tqdm(range(len(dataloader_size))):
        start_time_image = time.time()
        img, label = next(dataiter)
        
        if label[0] in ['s5','s6','s7']:
            actual.append('Unknown Class')
        else:
            actual.append(label[0])
        
        predicted.append(n_way_shot_learning(net, device, img, dict, threshold))
        end_time_image = time.time()
        avg_time_image.append(end_time_image-start_time_image)

    end_time = time.time()
    execution_time = end_time - start_time

    return classification_report(actual, predicted), execution_time, avg_time_image
