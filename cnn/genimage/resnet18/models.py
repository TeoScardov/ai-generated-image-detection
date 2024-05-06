import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class MultioutputResNet18(nn.Module):
    def __init__(self, baseline_model):
        super(MultioutputResNet18, self).__init__()
        self.features = nn.Sequential(*list(baseline_model.children())[:-1])
        self.fc1 = nn.Linear(512, 2)  # Output of size 2
        self.fc2 = baseline_model.fc  # Output of size 1000
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output_binary = self.fc1(x)
        output_multiclass = self.fc2(x)
        return output_binary, output_multiclass
    


def multiclass_finetuning():
    # Load the pretrained ResNet18 model
    pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Instantiate the custom model
    model = MultioutputResNet18(pretrained_model)

    return model



def multiclass_untrained():
    # Load the pretrained ResNet18 model
    model18 = resnet18()
    
    # Instantiate the custom model
    model = MultioutputResNet18(model18)

    return model
