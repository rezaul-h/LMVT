
from timm import create_model
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        self.model = create_model('inception_resnet_v2', pretrained=True)
        self.model.classif = nn.Linear(self.model.classif.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
