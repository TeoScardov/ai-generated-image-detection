import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights

def binary_finetuning():
    # Load the pretrained ResNet18 model
    model = resnet34(weights=ResNet34_Weights.DEFAULT)

    # Replace the last fully connected layer
    model.fc = nn.Linear(512, 2)

    return model



def binary_untrained():
    # Load the pretrained ResNet18 model
    model = resnet34()

    # Replace the last fully connected layer
    model.fc = nn.Linear(512, 2)

    return model



class MultioutputResNet34(nn.Module):
    def __init__(self, baseline_model):
        super(MultioutputResNet34, self).__init__()
        self.features = nn.Sequential(*list(baseline_model.children())[:-1])
        self.fc1 = nn.Linear(512, 2)  # Output of size 2
        self.fc2 = baseline_model.fc # Output of size 1000
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output_binary = self.fc1(x)
        output_multiclass = self.fc2(x)
        return output_binary, output_multiclass
    


def multiclass_finetuning():
    # Load the pretrained ResNet34 model
    pretrained_model = resnet34(weights=ResNet34_Weights.DEFAULT)
    
    # Instantiate the custom model
    model = MultioutputResNet34(pretrained_model)

    return model



def multiclass_untrained():
    # Load the empty ResNet34 model
    basic_model = resnet34()
    
    # Instantiate the custom model
    model = MultioutputResNet34(basic_model)

    return model