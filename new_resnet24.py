from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from torch.autograd import Variable
import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x

class resnet(nn.Module):
    __factory = {
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth=50, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(resnet, self).__init__()
        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in resnet.__factory:
            raise KeyError("Unsupported depth:", depth)
        depth = 50
        self.base = resnet.__factory[depth](pretrained=pretrained)
        self.features = list(self.base.children())


        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = True#norm
            self.dropout = -1#dropout
            self.has_embedding = False#num_features > 0
            self.num_classes = num_classes
          
            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            self.classifier = nn.Linear(512**2, 512)
            init.normal(self.classifier.weight, std=0.001)
            init.constant(self.classifier.bias, 0)
            self.classifier2 = nn.Linear(512, 8)
            init.normal(self.classifier2.weight, std=0.001)
            init.constant(self.classifier2.bias, 0)
            self.inconv1 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(1, 1), stride=1, padding=0,
                                     bias=True)
            self.inconv2 = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=1, padding=0,
                                     bias=True)
            self.inconv3 = nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=(1, 1), stride=1, padding=0,
                                     bias=True)
            self.inconv4 = nn.Conv2d(in_channels=2048, out_channels=8, kernel_size=(1, 1), stride=1, padding=0,
                                     bias=True)
            self.inconv1.cuda()
            self.inconv2.cuda()
            self.inconv3.cuda()
            self.inconv4.cuda()

            self.bn2 = torch.nn.BatchNorm2d(2)

            self.bn4 = torch.nn.BatchNorm2d(8)
            """self.base.layer2.conv1=self.inconv1
            self.base.layer2.bn1=self.bn2
            self.base.layer2.conv2=nn.Conv2d(2, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.base.layer2.conv1=self.inconv2
            self.base.layer2.bn1=self.bn2
            self.base.layer2.conv2=nn.Conv2d(2, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.base.layer3.conv1=self.inconv3
            self.base.layer3.bn1=self.bn2
            self.base.layer3.conv2=nn.Conv2d(2, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.base.layer4.conv1=self.inconv4
            self.base.layer4.bn1=self.bn4
            self.base.layer4.conv2=nn.Conv2d(2, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)"""

            self.base.fc=nn.Linear(in_features=2048, out_features=8, bias=True)
            self.stack1 = nn.Sequential(
                nn.Conv2d(512, 512,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512), nn.ReLU()
            )
            self.stack1_1 = nn.Sequential(
                nn.Conv2d(512, 512,
                          kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512), nn.ReLU()
            )
            self.stack3 = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512), nn.ReLU()
            )
        if not self.pretrained:
            self.reset_params()
        print(self)

    def cross_bilinear(self, a1, a2):
        Z = torch.bmm(a1, torch.transpose(a2, 1, 2)) / (7 ** 2)
        assert Z.size() == (a1.size(0), 512, 512)
        Z = Z.view(a1.size(0), 512 ** 2)
        Z = torch.sign(Z) * torch.sqrt(Z + 1e-5)
        Z = self.classifier(Z)
        Z = F.normalize(Z)

        return Z

    def forward(self, x):
        """res50_model = self.base
        handle1=res50_model.layer4.register_forward_hook(hook)
        handle2=res50_model.layer2.register_forward_hook(hook)

        for name,module in self.base.named_modules():
          print(x)
          x=module(x)"""
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break

            if name == 'layer3':
                l2 = Variable(x)

            x = Variable(module(x))
            l4 = Variable(x)

        """for name, param in self.base.named_parameters():
          print(name, param.size())

        res50_model = self.base
        res50_conv2 = ResNet50Bottom(res50_model)
        for i,child in enumerate(self.base.children()):
          print(i)
          if i==8:
            l4=x
            break
          if i==6:
            l2=x
          x=res50_conv2(x.detach())"""

        s2 = l2.sum(1) #/ 100
        #
        s4 = l4.sum(1) #/ 1000


        sw2 = s2 / (s2.view(x.size(0), -1).sum(1)).unsqueeze(1).unsqueeze(2)

        sw4 = s4 / (s4.view(x.size(0), -1).sum(1)).unsqueeze(1).unsqueeze(2)


        l2 = l2 * sw2.unsqueeze(1)
        l4 = l4 * sw4.unsqueeze(1)

        
        c2 = self.inconv2(l2)
        c4 = self.inconv4(l4)
        c2 = self.bn2(c2)
        c4 = self.bn4(c4)
        
        n2 = F.softmax(torch.mean(torch.mean(c2, dim=2), dim=2), dim=1)
        n4 = F.softmax(torch.mean(torch.mean(c4, dim=2), dim=2), dim=1)
        nn2 = n2.data.cpu().numpy()
        nn4 = n4.data.cpu().numpy()
        cam2 = np.zeros((x.size(0), 28, 28), dtype=float)
        cam4 = np.zeros((x.size(0), 7, 7), dtype=float)


        for i in range(0, x.size(0)):
            for j in range(0, 2):
                temp1 = c2[i, j, :, :].data.cpu().numpy()
                temp1 = np.maximum(temp1, 0)
                temp1 = temp1 - np.min(temp1)
                temp1 = temp1 / (np.max(temp1)+1e-8)
                cam2[i] = cam2[i] + nn2[i, j] * temp1
        cam2 = torch.FloatTensor(cam2)
        l2 = l2 * (cam2.unsqueeze(1).cuda())
        l2 = self.stack1(l2)
        l2 = self.stack1_1(l2)

        for i in range(0, x.size(0)):
            for j in range(0, 8):
                temp2 = c4[i, j, :, :].data.cpu().numpy()
                temp2 = np.maximum(temp2, 0)
                temp2 = temp2 - np.min(temp2)
                temp2 = temp2 / (np.max(temp2)+1e-8)
                cam4[i] =cam4[i] + nn4[i, j] * temp2
        cam4 = torch.FloatTensor(cam4)
        l4 = l4 * cam4.unsqueeze(1).cuda()
        l4 = self.stack3(l4)
        X = l2.view(x.size(0), 512, 7 ** 2)
        Y = l4.view(x.size(0), 512, 7 ** 2)
        Z = self.cross_bilinear(X, Y)
        return n2, n4, Z

    def get_embedding(self, x):
        return self.forward(x)
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
