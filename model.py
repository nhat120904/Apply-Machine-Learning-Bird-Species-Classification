import torch.nn as nn
from torchvision import models
    
class BirdClassifier(nn.Module):
    def __init__(self, architecture='resnet50', num_classes=200):
        super(BirdClassifier, self).__init__()
        if architecture == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif architecture == 'efficientnet_b5':
            self.model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
        elif architecture == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        elif architecture == 'efficientnet_b3':
            self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        elif architecture == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # replace the last fully connected layer
        if architecture.startswith('resnet'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif architecture in ['efficientnet_b5', 'efficientnet_b3', 'efficientnet_b0']:
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif architecture == 'mobilenet_v2':
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)