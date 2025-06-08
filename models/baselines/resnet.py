
import torchvision.models as models
import torch.nn as nn

def get_resnet18(num_classes=3):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet50(num_classes=3):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
