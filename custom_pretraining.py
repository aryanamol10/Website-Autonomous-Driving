# ==============================================================================
# COSMOS Capstone Project - Custom ResNet-18 Implementation Pipeline
# ------------------------------------------------------------------------------
# CREDITS & CITATIONS:
# This file implements the ResNet-18 deep convolutional architecture 
# introduced by He et al. (2015) in "Deep Residual Learning for Image Recognition".
# Base architectural weights are adapted from the open-source ImageNet-1K 
# checkpoint to enable high-efficiency feature extraction.
# ==============================================================================

import torch
import torch.nn as nn

# ==========================================
# EXPLICIT RESNET-18 ARCHITECTURE BLOCKS
# ==========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=5):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==========================================
# CONNECTING FUNCTION FOR TRAIN.PY
# ==========================================
def get_custom_pretrained_model(num_classes):
    """Instantiates our custom handwritten architecture and downloads the base 

    weights directly into it, popping off the old classification head.
    """
    print("📥 Custom module initializing weights from background repository...")
    
    # 1. Build the custom architecture structure
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
    # 2. Grab the raw parameters from the open-source hub link
    state_dict = torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet18-f37072fd.pth',
        progress=True
    )
    
    # 3. Pop the old fc layer weights so our custom 5-class output fits perfectly
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    
    # 4. Inject the weights into our custom code structure
    model.load_state_dict(state_dict, strict=False)
    return model