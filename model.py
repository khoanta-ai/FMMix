import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from orig_mixup import mixup_hidden
from features_aug import fm_mix_level, fmmix1, fmmix2, fmmix3, fmmix4

FM_augmentation_dict = {
    "mixup_hidden":mixup_hidden,
    'fm_mix_level': fm_mix_level,
    "fmmix1": fmmix1,
    "fmmix2": fmmix2,
    "fmmix3": fmmix3,
    "fmmix4": fmmix4,
}


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes, stride=1):
        super(PreActResNet, self).__init__()
        initial_channels = 64
        self.in_planes = initial_channels
        self.num_classes = num_classes

        self.args = args
        self.is_fm_mixup = self.args.is_fm_mixup
        
        if self.args.fm_mix_flag:
            self.ga_augment = FM_augmentation_dict[self.args.fm_augment_name]
        
        
        self.conv1 = nn.Conv2d(3,
                               initial_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(initial_channels * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def mixup_aug(self, x, target):
        if torch.rand(1).item() < self.args.p_fm_mix:
                if self.is_fm_mixup or self.args.mix_alg == "mixup_hidden":
                    out, computation_loss_components = self.ga_augment(args=self.args, x=x, 
                                                                                  target=target)                
        else: 
            if self.is_fm_mixup:
                raise Exception("Mixup switch is not implmented")
        return out, computation_loss_components


    def forward(self, x, fm_mix: bool = False, au_layer=None, target=None):
        out, computation_loss_components = None, None
        if self.args.rand_layers and fm_mix and (au_layer is None):
            au_layer = np.random.choice(self.args.choice_layers)
        
        out = x
        if au_layer == 0:
            if fm_mix:
                out, computation_loss_components = self.mixup_aug(x=out, target=target)    
        
        out = self.conv1(out)
        out = self.layer1(out)
        if au_layer == 1:
            if fm_mix:
                out, computation_loss_components = self.mixup_aug(x=out, target=target)    
        
        out = self.layer2(out)
        if au_layer == 2:
            if fm_mix:
                out, computation_loss_components = self.mixup_aug(x=out, target=target)    

        out = self.layer3(out)
        if au_layer == 3:
            if fm_mix:
                out, computation_loss_components = self.mixup_aug(x=out, target=target)    

        out = self.layer4(out)
        if au_layer == 4:
            if fm_mix:
                out, computation_loss_components = self.mixup_aug(x=out, target=target)    

        out = F.avg_pool2d(out, out.size(2))
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)

        
        return out, computation_loss_components
        

def ResNet18(args, num_classes, stride=1):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], args=args, num_classes=num_classes, stride=stride)

