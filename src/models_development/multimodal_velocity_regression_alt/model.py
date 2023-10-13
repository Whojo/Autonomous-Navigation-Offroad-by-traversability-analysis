import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18Velocity_Regression_Alt(nn.Module):
    def __init__(self, nb_classes=1, nb_input_features=1, nb_input_channels=7):
        super(ResNet18Velocity_Regression_Alt, self).__init__()

        self.nb_input_channels = nb_input_channels

        self.resnet18 = models.resnet18()

        # Replace the first convolutional layer to accept 7 channels
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=nb_input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # Replace the last fully-connected layer to have n classes as output
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 256)
        self.fc = nn.Linear(256, nb_classes)
        self.fc_speed = nn.Linear(1, self.resnet18.fc.in_features)

    def forward(
        self, x_img: torch.Tensor, x_dense: torch.Tensor
    ) -> torch.Tensor:
        x = x_img

        # Forward pass through the ResNet18
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        # x = torch.cat((x, x_dense), dim=1)
        x_speed = self.fc_speed(x_dense)
        x = x * x_speed

        x = self.resnet18.fc(x)
        x = F.relu(x)
        x = self.fc(x)
        x = torch.flatten(x)

        return x
