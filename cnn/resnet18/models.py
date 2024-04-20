import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def binary_feature_extraction_linear():
    # Load the pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer with a binary output
    model.fc = nn.Linear(512, 2)

    return model



def binary_feature_extraction_1hidden():
    # Load the pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer
    model.fc = nn.Sequential(
    nn.Linear(512, 64),  # Additional fc layer
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 2)  # Output layer
    )

    return model



def binary_finetuning():
    # Load the pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Replace the last fully connected layer
    model.fc = nn.Linear(512, 2)

    return model



def multiclass_feature_extraction_linear():
    # Load the pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    model.fc = nn.Linear(512, 20) 

    return model



def multiclass_feature_extraction_1hidden():
    # Load the pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    model.fc = nn.Sequential(
    nn.Linear(512, 128),  # Additional layer
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 20)  # Output layer
    )

    return model



def multiclass_feature_extraction_2hidden():
    # Load the pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    model.fc = nn.Sequential(
    nn.Linear(512, 128),  # Hidden layer 1
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),  # Hidden layer 2
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 20)  # Output layer
    )

    return model



def multiclass_finetuning():
    # Load the pretrained ResNet18 model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Replace the last fully connected layer
    model.fc = nn.Linear(512, 20)
    
    return model



def untrained_binary():
    # Load the ResNet18 model with random weights
    model = resnet18()

    # Replace the last fully connected layer
    model.fc = nn.Linear(512, 2)

    return model



def untrained_multiclass():
    # Load the ResNet18 model with random weights
    model = resnet18()

    # Replace the last fully connected layer
    model.fc = nn.Linear(512, 20)

    return model



class MultioutputResNet18(nn.Module):
    def __init__(self, pretrained_model):
        super(MultioutputResNet18, self).__init__()
        # Copy the model up to the last layer
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.fc1 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Output of size 2
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # Output of size 10
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        output_binary = self.fc1(x)
        output_multiclass = self.fc2(x)
        return output_binary, output_multiclass
    


def multioutput():
    # Load the pretrained ResNet18 model
    pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Instantiate the custom model
    model = MultioutputResNet18(pretrained_model)

    return model