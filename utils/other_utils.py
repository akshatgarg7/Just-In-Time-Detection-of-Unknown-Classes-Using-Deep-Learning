import os
from datetime import datetime
import numpy as np
import random
from PIL import Image

import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torch


def joinpath(rootdir, targetdir):
    """
    Joins the the rootdir and targetdir
    """
    # return os.path.join(os.sep, rootdir + os.sep, targetdir)
    return os.path.join(os.path.abspath(rootdir), targetdir)

def resultjoinpath(rootdir,resultdir,epochs,batch_size,loss_flag):
    _,result_name = os.path.split(rootdir)
    now = datetime.now()
    result_name = result_name + '_' + loss_flag + '_' + str(epochs) + '_' + str(batch_size) + '_' + now.strftime("%m_%d_%Y_%H_%M")
    return joinpath(resultdir, result_name)
    

"""
It will read two images and return them, as well as their label. 
If they are in the same category, i.e. the same person, 
it will return 0, and otherwise, it will return 1.
"""


class NetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,nchannel,flag,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.flag = flag
        self.nchannel = nchannel
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
    def __getitem__(self,index):
        if self.flag == "contrastive":
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
            if self.nchannel >=3:
                img0 = img0.convert("RGB")
                img1 = img1.convert("RGB")
            #Convert to grayscake
    #         img0 = img0.convert("L")
    #         img1 = img1.convert("L")

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
            
            return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
        
        elif self.flag == "triplet":
            anchor_img_tuple = random.choice(self.imageFolderDataset.imgs)
            anchor_label = anchor_img_tuple[1]
            
            while True:
                positive_img_tuple = random.choice(self.imageFolderDataset.imgs)
                positive_label = positive_img_tuple[1]
                if positive_label == anchor_label:
                    break
            
            while True:
                negative_img_tuple = random.choice(self.imageFolderDataset.imgs)
                negative_label = negative_img_tuple[1]
                if negative_label != anchor_label:
                    break

            anchor_img = Image.open(anchor_img_tuple[0])
            positive_img = Image.open(positive_img_tuple[0])
            negative_img = Image.open(negative_img_tuple[0])
            if self.nchannel >=3:
                anchor_img = anchor_img.convert("RGB")
                positive_img = positive_img.convert("RGB")
                negative_img = negative_img.convert("RGB")

            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img, anchor_label, positive_label, negative_label


"""
It will read one image at a time and return its label based on the folder it is present
"""

class TestNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,nchannel,flag,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.flag = flag
        self.nchannel = nchannel
        
    def __getitem__(self,index):
        img0_tuple = self.imageFolderDataset.imgs[index] # one image
        img0 = Image.open(img0_tuple[0])
        if self.nchannel >=3:
            img0 = img0.convert("RGB")

        # getting which classes represent which index
        d = {v: k for k, v in self.imageFolderDataset.class_to_idx.items()}

        # label returned with class name rather than class index
        label = d.get(img0_tuple[1])


        if self.transform is not None:
            img0 = self.transform(img0)
        
        return img0,label
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def get_dataset(path,nchannel,transforms,SiameseNetworkDataset,flag,num_workers,batch_size,shuffle):
    """
    This function gets the path and get dataset by applying transformation 
    and load the data using dataloader
    """
    folder_dataset = datasets.ImageFolder(root=path)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            nchannel=nchannel,
                                            flag=flag,
                                            transform=transforms)
    dataloader = DataLoader(siamese_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)
    return dataloader, siamese_dataset


def intersection(same,different):
    """
    This returns the number of intersection 
    between images belonging to same class and different class
    """
    no_of_intersection = []

    for i in same:
        if i > min(different):
            no_of_intersection.append(i)
    for j in different:
        if j < max(same):
            no_of_intersection.append(j)

    return no_of_intersection


def threshold(same,different,score):
    """
    Returns list of missclassified images based on the 
    given threshold score.
    """
    final_list = []
    same_misclassified = []
    different_misclassified = []
    for s in same:
        if s > score:
            final_list.append(s)
            same_misclassified.append(s)
    for d in different:
        if d <= score:
            final_list.append(d)
            different_misclassified.append(d)
    
    return final_list, same_misclassified, different_misclassified


def get_class_names(training_path):
    
    """
    Get all the training labels from train directory
    """
    
    class_names = []
    for entry in os.scandir(training_path):
        if entry.is_dir():
            class_names.append(entry.name)
    return class_names