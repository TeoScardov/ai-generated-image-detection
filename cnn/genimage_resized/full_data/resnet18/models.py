import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class MultioutputResNet18(nn.Module):
    def __init__(self, baseline_model):
        super(MultioutputResNet18, self).__init__()
        self.features = nn.Sequential(*list(baseline_model.children())[:-1])
        self.fc1 = nn.Linear(512, 1000)  # Output of size 1000 for each object class
        self.fc2 = nn.Linear(512, 9)  # Output of size 9 for each generator model (+natural)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output = {
            'multiclass': self.fc1(x),
            'generator': self.fc2(x)
        }
        return output 



def multiclass_finetuning():
    # Load the pretrained ResNet18 model
    pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Instantiate the custom model
    model = MultioutputResNet18(pretrained_model)

    return model