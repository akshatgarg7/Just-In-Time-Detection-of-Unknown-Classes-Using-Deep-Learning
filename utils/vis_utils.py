import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

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