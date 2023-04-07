# import torch
# import torch.nn as nn
# import torchvision.models as models

# class SiameseNetwork(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork, self).__init__()

#         # Load the pre-trained ResNet-18 model
#         resnet18 = models.resnet18(pretrained=True)

#         # Remove the last fully connected layer
#         modules = list(resnet18.children())[:-1]
#         self.resnet18 = nn.Sequential(*modules)

#         # Define the custom fully connected layers
#         self.fc1 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128)
#         )

#         # Freeze all layers in the model
#         for param in self.resnet18.parameters():
#             param.requires_grad = False

#         # Unfreeze the last layer of the last residual block
#         for param in self.resnet18[-2][-1].parameters():
#             param.requires_grad = True

#     def forward_once(self, x):
#         x = self.resnet18(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         return x

#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         return output1, output2
import torch
import torch.nn as nn
import torchvision.models as models
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet18 = models.resnet18(pretrained=True)

        # Remove the last two layers of the ResNet-18 model
        modules = list(resnet18.children())[:-2]
        self.resnet18 = nn.Sequential(*modules)

        # Define the custom fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        # Add the AdaptiveAvgPool2d and the fully connected layer back as a separate module
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc2 = nn.Linear(512, 128)

        # Freeze all layers in the model
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Unfreeze the last layer of the last residual block
        for param in self.resnet18[-1][-1].parameters():
            param.requires_grad = True

    def forward_once(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2