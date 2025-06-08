
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
