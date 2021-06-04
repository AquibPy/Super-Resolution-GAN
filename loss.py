import torch.nn as nn
from torchvision.models import vgg19_bn
import config

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19_bn(pretrained=True).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for params in self.vgg.parameters():
            params.requires_grad = False
    
    def forward(self,input,target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features,vgg_target_features)