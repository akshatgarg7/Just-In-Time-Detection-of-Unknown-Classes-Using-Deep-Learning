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

def imshow(img, text=None):
    """
    This is a helper function to show images
    """
    plt.figure()
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    
                         
def show_plot(iteration,loss):
    """
    This is a helper function to plot loss for any given run
    """
    plt.plot(iteration,loss)
    plt.show()

def example_vis(dataloader,flag):
    for i in range(5):
        if flag == 'contrastive':
            example_batch = next(iter(dataloader))

            # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
            # If the label is 1, it means that it is not the same person, label is 0, same person in both images
            concatenated = torch.cat((example_batch[0], example_batch[1]),0)

            imshow(torchvision.utils.make_grid(concatenated))
            print(example_batch[2].numpy().reshape(-1))
        elif flag == 'triplet':
            example_batch = next(iter(dataloader))
            images = torch.cat((example_batch[0], example_batch[1], example_batch[2]), dim=0)
            # display the concatenated images
            imshow(torchvision.utils.make_grid(images, nrow=10))