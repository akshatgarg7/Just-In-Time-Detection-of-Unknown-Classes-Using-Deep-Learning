from collections import defaultdict

def create_embeddings(net,device,dataloader,dataloader_size):
    
    """
    This function returns a dict with all the 
    embeddings of dataloader
    """
    
    embeddings_dict = defaultdict(list)
    dataiter = iter(dataloader)

    for i in range(len(dataloader_size)):
        img,label = next(dataiter)
        img = img.to(device)
        embeddings_dict[label[0]].append(net.forward_once(img))

    return embeddings_dict