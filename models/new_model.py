# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# import torchvision.utils
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F

# # New architecture, TODO: still in testing
# class NewSiameseNetwork(nn.Module):
#     def __init__(self,nchannel):
#         super(NewSiameseNetwork, self).__init__()
#         self.nchannel = nchannel
#         # Setting up the Sequential of CNN Layers
#         # self.cnn1 = nn.Sequential(
#         #     nn.Conv2d(nchannel, 96, kernel_size=5, stride=2,),
#         #     nn.BatchNorm2d(96),
#         #     nn.ReLU(inplace=True),
#         #     nn.MaxPool2d(2, stride=2),

#         #     nn.Conv2d(96, 256, kernel_size=3, stride=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=True),
#         #     nn.MaxPool2d(2, stride=2),

#         #     nn.Conv2d(256, 384, kernel_size=3, stride=1),
#         #     nn.BatchNorm2d(384),
#         #     nn.ReLU(inplace=True),

#         #     nn.Conv2d(384, 512, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(inplace=True),
#         #     nn.MaxPool2d(2, stride=2),

#         #     nn.Conv2d(512, 256, kernel_size=3, stride=1),
#         #     nn.ReLU(inplace=True)

#         # )

#         # self.fc1 = nn.Sequential(
            
#         #     nn.Linear(4096,2048),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(0.2),


#         #     nn.Linear(2048,1024),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(0.2),


#         #     nn.Linear(1024,526),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(0.2),

            
#         #     nn.Linear(526, 256),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(0.2),


#         #     nn.Linear(256,128)
#         # )
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(nchannel, 96, kernel_size=7, stride=2),  #200*200
#             nn.BatchNorm2d(96),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(96, 256, kernel_size=5, stride=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(256, 384, kernel_size=3, stride=1),
#             nn.BatchNorm2d(384),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(384, 512, kernel_size=3, stride=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(512, 256, kernel_size=3, stride=1),
#             nn.ReLU(inplace=True)
#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(12544, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),

#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),

#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),

#             nn.Linear(256, 128)
#         )


#     def forward_once(self, x):
#         x = self.cnn1(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         return x


#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         return output1, output2



import torch.nn as nn

# New architecture, TODO: still in testing
class NewSiameseNetwork(nn.Module):
    def __init__(self,nchannel):
        super(NewSiameseNetwork, self).__init__()
        self.nchannel = nchannel
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(nchannel, 96, kernel_size=5, stride=2,),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(96, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(512, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)

        )

        self.fc1 = nn.Sequential(

            nn.Linear(1024,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
 
            nn.Linear(256,128)
        )
    def forward_once(self, x):
        x = self.cnn1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2