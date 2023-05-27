import time
import os
import torchvision.transforms as transforms
from PIL import Image
import torch
from models import pretrained_model
import torch.nn.functional as F
import argparse
import pickle
# from utils.other_utils import get_dataset,TestNetworkDataset
# from embeddings.n_way_shot_learning_mean import n_way_shot_learning

def joinpath(rootdir, targetdir):
    """
    Joins the the rootdir and targetdir
    """
    # return os.path.join(os.sep, rootdir + os.sep, targetdir)
    return os.path.join(os.path.abspath(rootdir), targetdir)

def n_way_shot_learning(net, device, image, embedding_dict, threshold):
    
    output = net.forward_once(image.to(device))
    # print(output)
    
    labels = list(embedding_dict.keys())
    embedding_tensors = torch.stack(list(embedding_dict.values()))
    
    scores = F.pairwise_distance(output, embedding_tensors)
    # print(scores)
    dict_results = dict(zip(labels,scores))

    dict_results_threshold = {k:-v for k,v in dict_results.items() if v < threshold}
    if len(dict_results_threshold) == 0:
        return "Unknown Class"
    values = torch.tensor(list(dict_results_threshold.values()))
    values_softmax = F.softmax(values)
    
    final_dict = {k:v.item() for k,v in zip(dict_results_threshold.keys(),values_softmax)}
    return max(final_dict,key=final_dict.get)

def transformations():
    transformation = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] 
                                    )
    return transformation


def get_image(path,transformation):
    # Load the image from file
    image = Image.open(path)
    print(type(image))

    # Apply the transformation to the image
    transformed_image = transformation(image)

    # Add a batch dimension to the transformed image
    transformed_image = transformed_image.unsqueeze(0)

    # Create a PyTorch tensor from the transformed image
    tensor_image = torch.Tensor(transformed_image)

    return tensor_image

def load_model(device,MODEL_PATH):
    net = pretrained_model.SiameseNetwork().to(device)
    net.load_state_dict(torch.load(joinpath(MODEL_PATH,'model.pt')))
    net.eval()
    return net

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # MODEL_PATH = "saved_model/lab_parts_100_64_04_26_2023_02_52"
    # net = load_model(device,MODEL_PATH)
    # with open(joinpath(MODEL_PATH,'mean_dict.pkl'),'rb') as file:
    #     dict = pickle.load(file)
    # # print(dict)
    # threshold = 0.59
    for i in range(1,26):
        img = get_image('data/lab_parts/test/6.7L Ford/fuelinjector_6.7L Ford_camC_' + str(i) + '.jpg',transformations())
    # print(img)
    # test_dataloader, test_dataset = get_dataset('data/lab_parts/test', 3, transformations(), 
    #                                         TestNetworkDataset,'contrastive', 0, 1, False)
    # dataiter = iter(test_dataloader)

    # img,_ = next(dataiter)
    # print(img)
    # print('-----------------------------------------------')
        # label = n_way_shot_learning(net,device,img,dict,threshold)
        # print(label)

main()

# if '__name__' == '__main__':
#     parser = argparse.ArgumentParser(description="deployment part")
#     # Adding required argument
#     parser.add_argument("path", type = str, help = "Path where model and other variables saved")
#     parser.add_argument("-c", "--cuda", default = False, type = bool, 
#                         help = "Enables CUDA training (default: False)")
    
#     global args, device
#     # Read arguments from command line
#     args = parser.parse_args()
#     args.cuda = args.cuda and torch.cuda.is_available()

#     if args.cuda:
#         device = 'cuda'
#     else:
#         device = 'cpu'

#     main()