
import torchvision.models as models
import torch.nn as nn

def get_vit_b_16(num_classes=3):
    model = models.vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
