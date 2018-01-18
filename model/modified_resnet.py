import torch
import torch.nn as nn
import torchvision

class Modified_ResNet18(nn.Module):

    def __init__(self, base_model=None,  num_classes=None):
        super(Modified_ResNet18, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1001, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, size):        
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc(x)
        
        size = size.unsqueeze(1)
        x = torch.cat((x, size),1)
        x = self.fc(x)
        return x
    
def modified_resnet18(num_classes):
    
    model = Modified_ResNet18(base_model=torchvision.models.resnet.resnet50(pretrained=True), 
                              num_classes=num_classes)
    return model