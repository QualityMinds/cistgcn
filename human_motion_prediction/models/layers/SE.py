import torch.nn as nn


# https://github.com/MotionMLP/MotionMixer/blob/main/h36m/mlp_mixer.py
class SELayer1d(nn.Module):
    def __init__(self, channel, reduction=4, use_max_pooling=False):
        super(SELayer1d, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=4, use_max_pooling=False):
        super(SELayer2d, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) if not use_max_pooling else nn.AdaptiveMaxPool2d(1)
        res = channel // reduction
        if res < 1: res = 1
        self.excitation = nn.Sequential(
            nn.Linear(channel, res, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(res, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
