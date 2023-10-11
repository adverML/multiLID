'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, preprocessing={}, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.mu = None
        self.sigma = None

        self.preprocessing = preprocessing
        
        if not len(self.preprocessing) == 0:
            self.mu = torch.tensor(preprocessing['mean']).float().view(3, 1, 1).cuda()
            self.sigma = torch.tensor(preprocessing['std']).float().view(3, 1, 1).cuda()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        if not len(self.preprocessing) == 0:
            # print("ResNet Sota Net Normalization")
            x = (x - self.mu) / self.sigma

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)  # logit values that go into the softmax or log-softmax
        return out
    
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        pen = self.layer4(out)
        out = F.avg_pool2d(pen, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)  # logit values that go into the softmax or log-softmax
        return out, pen

    def intermediate_forward(self, x, layer_index):
        if layer_index == 0:
            return x

        x = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            return x

        x = self.layer1(x)
        if layer_index == 2:
            return x

        x = self.layer2(x)
        if layer_index == 3:
            return x

        x = self.layer3(x)
        if layer_index == 4:
            return x

        x = self.layer4(x)
        if layer_index == 5:
            return x

        x = F.avg_pool2d(x, 4)
        if layer_index == 6:
            return x

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if layer_index == 7:
            return x

    def layer_wise(self, x):
        # Method to get the layer-wise embeddings for the proposed method
        # Input is included as the first layer
        output = [x] #1
        out = F.relu(self.bn1(self.conv1(x)))
        output.append(out) #2
        out = self.layer1(out)
        output.append(out) #3
        out = self.layer2(out)
        output.append(out) #4
        out = self.layer3(out)
        output.append(out) #5
        out = self.layer4(out)
        output.append(out) #6
        out = F.avg_pool2d(out, 4)
        output.append(out) #7
        out = out.view(out.size(0), -1)
        # output.append(out)
        out = self.linear(out)
        output.append(out) #8 (logits)

        return output

    def layer_wise_deep_mahalanobis(self, x):
        # Method to get the layer-wise embeddings for the deep mahalanobis detection method
        # Input is included as the first layer
        output = [x] #1
        out = F.relu(self.bn1(self.conv1(x)))
        output.append(out) #2
        out = self.layer1(out)
        output.append(out) #3
        out = self.layer2(out)
        output.append(out) #4
        out = self.layer3(out)
        output.append(out) #5
        out = self.layer4(out)
        output.append(out) #6
        out = F.avg_pool2d(out, 4)
        output.append(out) #7
        out = out.view(out.size(0), -1)
        # output.append(out)
        out = self.linear(out)
        output.append(out) #8 (logits)

        return out, output

    def layer_wise_odds_are_odd(self, x):
        # Method to get the latent layer and logit layer outputs for the "odds-are-odd" method
        output = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        output.append(out)  # latent
        out = self.linear(out)
        output.append(out)  # logits

        return output

    def layer_wise_lid_method(self, x):
        # Method to get the layer-wise embeddings for the LID adversarial subspaces paper
        # Input is included as the first layer
        output = [x] #1
        out = self.conv1(x)
        output.append(out)  #2
        out = self.bn1(out)
        output.append(out)  #3
        out = F.relu(out)
        output.append(out)  #4
        out = self.layer1(out)
        output.append(out)  #5
        out = self.layer2(out)
        output.append(out)  #6
        out = self.layer3(out)
        output.append(out)  #7
        out = self.layer4(out)
        output.append(out)  #8
        out = F.avg_pool2d(out, 4)
        output.append(out)  #9
        out = out.view(out.size(0), -1)
        # output.append(out)
        out = self.linear(out)
        output.append(out)  #10 (logits)

        return output


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34_SOTA(preprocessing={}, num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], preprocessing=preprocessing,  num_classes=num_classes)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
