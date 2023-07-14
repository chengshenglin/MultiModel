import torch
import torch.nn as nn
from torchvision.models import resnet50


class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.image_res = resnet50(pretrained=True)
        self.resnet = nn.Sequential(
            *(list(self.image_res.children())[:-1]),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.image_res.fc.in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1000)
        )
        
        for param in self.image_res.parameters():
            param.requires_grad = True

    def forward(self, imgs):
        feature = self.resnet(imgs)
        x = self.fc(feature)
        return x

