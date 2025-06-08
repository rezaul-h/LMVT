
import timm
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
