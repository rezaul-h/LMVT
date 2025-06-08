
import torchvision.models as models
import torch.nn as nn

def get_densenet121(num_classes=3):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
