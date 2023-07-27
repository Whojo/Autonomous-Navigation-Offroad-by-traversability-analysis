import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import params

# Import custom packages
import params.learning

NORMALIZE_PARAMS = params.learning.NORMALIZE_PARAMS

class ResNet18Velocity_Regression_Alt(nn.Module):
    
    def __init__(self,
                 nb_classes=1,
                 nb_input_features=1,
                 nb_input_channels=7):
        
        super(ResNet18Velocity_Regression_Alt, self).__init__()
        
        self.nb_input_channels = nb_input_channels
        
        ## Image and depth image processing ##
        # Load the ResNet18 model with pretrained weights
        self.resnet18 = models.resnet18()
        # self.resnet18 = models.resnet18(
        #     weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace the first convolutional layer to accept 7 channels
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=nb_input_channels,
            out_channels=64,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        # Replace the last fully-connected layer to have n classes as output
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 256)
        self.fc = nn.Linear(256, nb_classes)
        
        ## Numeric input processing ##
        # self.mlp = nn.Sequential(
        #     nn.Linear(num_input_features,
        #               params.learning.IMAGE_SHAPE[1]*params.learning.IMAGE_SHAPE[0]*7//2),
        #     nn.ReLU(),
        #     nn.Linear(params.learning.IMAGE_SHAPE[1]*params.learning.IMAGE_SHAPE[0]*7//2,
        #               params.learning.IMAGE_SHAPE[1]*params.learning.IMAGE_SHAPE[0]*7)
        # )
        
        # self.fc = nn.Linear(
        #     nb_input_features,
        #     params.learning.IMAGE_SHAPE[1]*
        #     params.learning.IMAGE_SHAPE[0]*
        #     nb_input_channels)
        
        # print(params.learning.IMAGE_SHAPE[1]*params.learning.IMAGE_SHAPE[0]*7)
        # print(params.learning.IMAGE_SHAPE[1]*params.learning.IMAGE_SHAPE[0]*7//2)
        
    
    def forward(self,
                x_img: torch.Tensor,
                x_dense: torch.Tensor) -> torch.Tensor:
        
        # FC layer to convert the numeric input to the same shape as the
        # activation map
        # x_dense = self.fc(x_dense)
        # x_dense = x_dense.view(-1,
        #                        self.nb_input_channels,
        #                        params.learning.IMAGE_SHAPE[0],
        #                        params.learning.IMAGE_SHAPE[1]) 
        
        # Element-wise product of the activation map and the main-
        # channel input
        # x = x_img * x_dense
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
        
        # print(x.shape)
        x = F.avg_pool2d(x, x.size()[2:])
        # print(x.shape)
        x = x.view(x.size(0), -1)
        
        #x = torch.cat((x, x_dense), dim=1)
        # print(x.shape)
        
        x = self.resnet18.fc(x)
        x = x * x_dense

        x = F.relu(x)
        
        x = self.fc(x)

        x = torch.flatten(x)
        
        return x
    
test_transform = transforms.Compose([
    # transforms.Resize(100),
    transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
    # transforms.Grayscale(),
    # transforms.CenterCrop(100),
    # transforms.RandomCrop(100),
    transforms.ToTensor(),
    
    # Mean and standard deviation were pre-computed on the training data
    # (on the ImageNet dataset)
    transforms.Normalize(
        mean=NORMALIZE_PARAMS["rbg"]["mean"],
        std=NORMALIZE_PARAMS["rbg"]["std"]
    ),
])

transform_depth = transforms.Compose([
        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),
        
        # Reduce the size of the images
        # (if size is an int, the smaller edge of the
        # image will be matched to this number and the ration is kept)
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        
        # Normalize a tensor image with pre-computed mean and standard deviation
        # (based on the data used to train the model(s))
        # (be careful, it only works on torch.*Tensor)
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["depth"]["mean"],
            std=NORMALIZE_PARAMS["depth"]["std"]
        ),
])

transform_normal = transforms.Compose([
        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),
        
        # Reduce the size of the images
        # (if size is an int, the smaller edge of the
        # image will be matched to this number and the ration is kept)
        transforms.Resize(params.learning.IMAGE_SHAPE, antialias=True),
        
        # Normalize a tensor image with pre-computed mean and standard deviation
        # (based on the data used to train the model(s))
        # (be careful, it only works on torch.*Tensor)
        transforms.Normalize(
            mean=NORMALIZE_PARAMS["normal"]["mean"],
            std=NORMALIZE_PARAMS["normal"]["std"]
        ),
])