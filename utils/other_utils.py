import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


def joinpath(rootdir, targetdir):
    """
    Joins the the rootdir and targetdir
    """
    return os.path.join(os.sep, rootdir + os.sep, targetdir)



class SiameseNetworkDataset(Dataset):   
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        #Convert to grayscake
#         img0 = img0.convert("L")
#         img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetworkDataset_for_test(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = self.imageFolderDataset.imgs[index] # one image
        img0 = Image.open(img0_tuple[0])

        # getting which classes represent which index
        d = {v: k for k, v in self.imageFolderDataset.class_to_idx.items()}

        # label returned with class name rather than class index
        label = d.get(img0_tuple[1])


        if self.transform is not None:
            img0 = self.transform(img0)
        
        return img0,label
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

def get_dataset(path,transforms,SiameseNetworkDataset,num_workers,batch_size,shuffle):
    folder_dataset = datasets.ImageFolder(root=path)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms)
    dataloader = DataLoader(siamese_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def intersection(same,different):
    no_of_intersection = []

    for i in same_disssimilarity:
        if i > min(different_disssimilarity):
            intersection.append(i)
    for j in different_disssimilarity:
        if j < max(same_disssimilarity):
            intersection.append(j)

    return no_of_intersection