import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class HCCModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(HCCModel, self).__init__()
        
        if pretrained:
            # Use pre-trained MobileNetV2 (lighter than ResNet18)
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            # Modify first conv layer to accept grayscale
            self.backbone.features[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            # Modify classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        else:
            # Simple CNN (original design, slightly improved)
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128 * 28 * 28, 256)
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if hasattr(self, 'backbone'):
            return self.backbone(x)
        else:
            # Simple CNN forward
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
