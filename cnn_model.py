import torch
import torch.nn as nn

class StreetAidCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(StreetAidCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
        self.regressor = nn.Linear(128, 4)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        class_out = self.classifier(x)
        coord_out = self.regressor(x)
        return class_out, coord_out