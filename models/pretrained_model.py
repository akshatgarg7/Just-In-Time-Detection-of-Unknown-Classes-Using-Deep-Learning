import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):

    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()

        # Remove the last fully connected layer from the VGG16 model
        self.features = base_model.features
        self.classifier = nn.Sequential(*list(base_model.classifier[:-1]))

        # Add custom fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


