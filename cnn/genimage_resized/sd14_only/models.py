import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class MultioutputResNet50(nn.Module):
    def __init__(self, baseline_model):
        super(MultioutputResNet50, self).__init__()
        self.features = nn.Sequential(*list(baseline_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 2)  # Output of size 2
        self.fc2 = nn.Linear(2048, 1000)  # Output of size 1000
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output_binary = self.fc1(x)
        output_multiclass = self.fc2(x)
        return output_binary, output_multiclass



def multiclass_finetuning():
    # Load the pretrained ResNet50 model
    pretrained_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Instantiate the custom model
    model = MultioutputResNet50(pretrained_model)

    return model



def multiclass_untrained():
    # Load the empty ResNet50 model
    basic_model = resnet50()
    
    # Instantiate the custom model
    model = MultioutputResNet50(basic_model)

    return model



def multiclass_finetuning_1hidden():
    # Load the pretrained ResNet50 model
    pretrained_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Instantiate the custom model
    model = MultioutputResNet50(pretrained_model)
    model.fc1 = nn.Sequential(
                            nn.Linear(2048, 256),  # Additional fc layer
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(256, 2)  # Output of size 2
                            )
    model.fc2 = nn.Sequential(
                            nn.Linear(2048, 2048),  # Additional fc layer
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(2048, 1000)  # Output of size 1000
                            )

    return model
