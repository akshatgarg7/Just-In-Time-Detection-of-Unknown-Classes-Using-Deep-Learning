import os
import argparse
import pickle
from tqdm import tqdm
import warnings
import plotly.graph_objects as go

import torchvision.transforms as transforms
import torch
from torch import optim


from models.old_model import SiameseNetwork
from models.new_model import NewSiameseNetwork
from models import pretrained_model

from losses.euclidean import ContrastiveLoss
from losses.cosine import ContrastiveLossCosine
from losses.triplet import TripletLoss

from utils.vis_utils import example_vis
from utils.other_utils import joinpath, resultjoinpath, NetworkDataset, get_dataset


# Suppress all warnings
warnings.filterwarnings("ignore")

def transformations(rotation=False):
    if rotation:
        transformation = transforms.Compose([transforms.Resize((224,224)),
            transforms.ToTensor(), 
            transforms.RandomApply([
                transforms.RandomRotation([-30,30])], p = 0.2)
        ])
    else:
        # transformation = transforms.Compose([transforms.Resize((100,100)),
        #                                  transforms.ToTensor()])
        transformation = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] 
                                        )
    return transformation

def contrastive_train(net,optimizer,criterion,epochs,train_dataloader,valid_dataloader):
    loss_history = []
    valid_loss_history = []
    for epoch in tqdm(range(epochs), desc='Epochs'):
        # Iterate over batches
        # for (img0, img1, label) in train_dataloader:
        for step,((img0, img1, label),(valid_img0,valid_img1,valid_label)) in enumerate(zip(train_dataloader,valid_dataloader),0):

            # Send the images and labels to CUDA
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            # print(img0.shape)
            valid_img0, valid_img1, valid_label = valid_img0.to(device), valid_img1.to(device), valid_label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)
            valid_output1, valid_output2 = net(valid_img0, valid_img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)
            valid_loss_contrastive = criterion(valid_output1, valid_output2, valid_label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()
        # save the loss history for each epoch
        loss_history.append(loss_contrastive.item())
        valid_loss_history.append(valid_loss_contrastive.item())
    return(loss_history,valid_loss_history)

def triplet_train(net,optimizer,criterion,epochs,train_dataloader,valid_dataloader):
    loss_history = []
    valid_loss_history = []
    for epoch in tqdm(range(epochs), desc='Epochs'):
        # Iterate over batches
        for step, ((anchor_img, positive_img, negative_img,_,_,_),(anchor_valid_img, positive_valid_img, negative_valid_img,_,_,_)) in enumerate(zip(train_dataloader,valid_dataloader),0):
            # Send the images and labels to CUDA
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)
            anchor_valid_img, positive_valid_img, negative_valid_img = anchor_valid_img.to(device), positive_valid_img.to(device), negative_valid_img.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Pass in the three images into the network and obtain three outputs
            anchor_out = net.forward_once(anchor_img)
            positive_out = net.forward_once(positive_img)
            negative_out = net.forward_once(negative_img)

            anchor_valid_out = net.forward_once(anchor_valid_img)
            positive_valid_out = net.forward_once(positive_valid_img)
            negative_valid_out = net.forward_once(negative_valid_img)
            
            # Pass the outputs of the networks and label into the loss function
            loss_triplet = criterion(anchor_out, positive_out, negative_out)
            loss_valid_triplet = criterion(anchor_valid_out, positive_valid_out, negative_valid_out)

            # Calculate the backpropagation
            loss_triplet.backward()

            # Optimize
            optimizer.step()
        # save the loss history for each epoch
        loss_history.append(loss_triplet.item())
        valid_loss_history.append(loss_valid_triplet.item())
    return loss_history,valid_loss_history

def main():

    if device == "cuda":
        print('Training on: ',torch.cuda.get_device_name())
        torch.cuda.empty_cache()
    else:
        print('Training on: ',device)
    
    PATH = args.path
    nchannel = args.n_channel
    FLAG = args.loss_flag
    BATCH_SIZE = args.batch_size
    MODEL_PATH = resultjoinpath(PATH,args.save_dir,args.epochs,BATCH_SIZE,args.loss_flag)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    print("---------------------------------------------------------------------------------------")
    print("Starting the training process")
    train_dataloader,train_size = get_dataset(joinpath(PATH,'train'), nchannel, transformations(args.rotation),NetworkDataset,FLAG,0,BATCH_SIZE,True)
    valid_dataloader,valid_size = get_dataset(joinpath(PATH,'valid'), nchannel, transformations(args.rotation),NetworkDataset,FLAG,0,BATCH_SIZE,True)
    
    net = pretrained_model.SiameseNetwork().to(device)
    # net = NewSiameseNetwork().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if FLAG == 'contrastive':
        criterion = ContrastiveLoss()
        # criterion = ContrastiveLossCosine()
    elif FLAG == 'triplet':
        criterion = TripletLoss()

    # print(net)

    if FLAG == 'contrastive':
        train_loss, valid_loss= contrastive_train(net, optimizer, criterion, args.epochs, train_dataloader, valid_dataloader)
        # _, train_loss = train()
    elif FLAG == 'triplet':
        train_loss, valid_loss = triplet_train(net, optimizer, criterion, args.epochs, train_dataloader, valid_dataloader)
        # Make sure to also calculate valid_loss for the triplet case
    print("training completed")
    print("---------------------------------------------------------------------------------------")

    with open(joinpath(MODEL_PATH,'loss_list.pkl'), 'wb') as f:
    # pickle the two lists
        pickle.dump(train_loss, f)
        pickle.dump(valid_loss, f)
    torch.save(net.state_dict(), joinpath(MODEL_PATH,'model.pt'))

    # Create the figure
    fig = go.Figure()   

    # Add traces for train_loss and valid_loss
    fig.add_trace(go.Scatter(x=list(range(len(train_loss))), y=train_loss,
                        mode='lines', name='Training Loss', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=list(range(len(valid_loss))), y=valid_loss,
                        mode='lines', name='Validation Loss', line=dict(color='blue'))) 

    # Set the title and axis labels
    fig.update_layout(title='Loss vs Epochs', xaxis_title='Epoch', yaxis_title='Loss')  

    # Display the figure
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the network and save the model")

    # Adding required argument
    parser.add_argument("path", type = str, help = "Path where the data is located")

    # Adding optional argument
    parser.add_argument("--n_channel", default = 3, type = int, 
                        help = "Number of channels in images (default: 3)")
    parser.add_argument("-e", "--epochs", default = 100, type = int, 
                        help = "Number of epochs (default: 100)")
    parser.add_argument("--save_dir", default = "saved_model", type = str, 
                        help = "Directory to store model and losses (default: '/saved_model')")
    parser.add_argument("-c", "--cuda", default = False, type = bool, 
                        help = "Enables CUDA training (default: False)")
    parser.add_argument("-r", "--rotation", default = False, type = bool, 
                        help = "Enables rotation of the images (default: False)")
    parser.add_argument("-f", "--loss_flag", default = "contrastive", type = str, 
                        help = "Set the kind of loss (default: 'contrastive')")
    parser.add_argument("-l", "--lr", default = 0.0005, type = float, 
                        help = "learning rate (default: 0.0005)")
    parser.add_argument("-b", "--batch_size", default = 64, type = int, 
                        help = "input batch size for training (default: 64)")
    # parser.add_argument("--name_result", default = None, type = str, 
    #                     help = "result name where to save (default: None)")

    global args, device
    # Read arguments from command line
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    main()