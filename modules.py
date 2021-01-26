'''
Face mask classification model using mobilenet
The model originally aim to deploy on edge device
'''

from __future__ import print_function
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, models, transforms
import sys
import torchvision.models as models


def set_parameter_requires_grad(model, feature_extracting):
	if feature_extracting:
		for param in model.parameters():
			param.requires_grad = False


class FaceMask_CNN(torch.nn.Module):

    def __init__(self, n_class, use_pretrained, feature_extract):

        super(FaceMask_CNN, self).__init__()

        input_channel = 32
        input_channel = _make_divisible(input_channel * 1.0, 8)

        self.backbone = models.mobilenet_v2(pretrained=False)
        set_parameter_requires_grad(self.backbone, feature_extract)
        self.backbone.features[0] = ConvBNReLU(1, input_channel, stride=2, norm_layer=nn.BatchNorm2d)

        self.backbone.classifier  = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, n_class),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.backbone(x)
        return x


# code is taken from pytorch mobilenet v2
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

