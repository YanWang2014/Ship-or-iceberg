import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math

class Vgg11bnNet(nn.Module):
    def __init__(self, model_vgg, num_classes):
        super(Vgg11bnNet, self).__init__()
        self.features = nn.Sequential(*list(model_vgg.features.children()))# 28 is total
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3,1,padding=1),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.6),
            nn.BatchNorm2d(512),
            #nn.MaxPool2d(kernel_size=2,stride=1),#7*7->6*6
            
            nn.Conv2d(512, 128, 3,1,padding=1),
            nn.ReLU(inplace=True),

            
        )
        self.fc = nn.Linear(129,num_classes) #need to be considered
        self.dropout=nn.Dropout()
        self._initialize_weights()
        
    def forward(self, x,size):
        x = self.features(x)
        x = self.classifier(x)
        r = x.size(3)       
        x = F.avg_pool2d(x, r)
        x = x.view(x.size(0), -1)
        
        size = size.unsqueeze(1)
        x = torch.cat((x,size),1)
        x = self.fc(F.relu(x))
        return x
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()  

def modified_vgg11(num_classes):
    
    model = Vgg11bnNet(model_vgg=torchvision.models.vgg19_bn(pretrained=True), 
                              num_classes=num_classes)
    return model
