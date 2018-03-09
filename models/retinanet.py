import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fpn import *
from layers import conv3x3, conv1x1


class RetinaNet(nn.Module):
    anchors = 9
    backbones = {
        'resnet50': fpn50,
        'resnet101': fpn101,
        'resnet152': fpn152
    }

    def __init__(self, backbone='resnet101', pretrained=True, num_classes=21):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = self.backbones[backbone](pretrained=pretrained)
        self.classifier = nn.Sequential(
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, self.num_classes * self.anchors, stride=1, padding=1),
            #nn.Sigmoid(),

        )
        self.regressor = nn.Sequential(
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, 256, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(256, 4 * self.anchors, stride=1, padding=1),
        )

        self.__init__classfier_output((list(self.classifier.children())[-1]).bias.data)

    def __init__classfier_output(self, tensor, pi=0.01):
        fill_constant = -1 * math.log((1 - pi) / pi)
        if isinstance(tensor, Variable):
            self.__init_classifier_output(tensor.data, pi)

        return tensor.fill_(fill_constant)

    def forward(self, x):
        pred_cls = []
        pred_loc = []
        fms = self.backbone(x)
        for fm in fms:
            cls_pred = self.classifier(fm)
            loc_pred = self.regressor(fm)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()  # [n, c, h, w] => [n, h, w, c]
            cls_pred = cls_pred.view(x.size(0), -1, self.num_classes)  # [n, h, w, c] => [n, h * w * anchors, num_classes]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(x.size(0), -1, 4)
            pred_cls.append(cls_pred)
            pred_loc.append(loc_pred)

        return torch.cat(pred_loc, 1), torch.cat(pred_cls, 1)


def retinanet50(pretrained=True, num_classes=21):
    model = RetinaNet('resnet50', pretrained, num_classes)
    return model


def retinanet101(pretrained=True, num_classes=21):
    model = RetinaNet('resnet101', pretrained, num_classes)
    return model
