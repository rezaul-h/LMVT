
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, num_classes=3):
        super(Model, self).__init__()
        self.model = models.squeezenet1_1(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)
