import numpy as np
import torch
from torch import nn
from torch.nn import init


class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 通道拆分融合
class Reducedim1(nn.Module):
    def __init__(self, channel=431, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.mlplayers = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1),# [bs, N, 16, 16]
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1,stride=2),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1),# [bs, N, 8, 8]
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1,stride=2),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, 256, kernel_size=(3, 3), padding=1),# [48, 431, 4, 4]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)# [48, 256, 3, 3]
        )
    def forward(self, feaature):
        y = feaature # [bs, 256, N]
        y = y.view(y.shape[0], 16, 16, y.shape[2]) # [bs, 16, 16, N]
        y = y.permute(0,3,1,2)
        y = self.mlplayers(y)

        y = y.contiguous().view(y.shape[0], y.shape[1], -1)

        y = y.view(y.shape[0], -1)
        return y

# N补零降维
class Reducedim2(nn.Module):
    def __init__(self, channel=256, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.mlplayers = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1),# [bs, 256, 21, 21]
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1,stride=2),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1),# [48, 256, 11, 11]
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1,stride=2),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1),# [48, 256, 6, 6]
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1,stride=2),
            nn.BatchNorm2d(channel),
            nn.ReLU(True) # [48, 256, 3, 3]
        )
    def forward(self, feature):
        '''
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        # [bs, 256, 431]
        zero = torch.empty(feature.shape[0], feature.shape[1], 10).to(torch.device('cuda'))
        torch.zeros_like(zero)
        y = feature
        if feature.shape[2]==431 : y = torch.cat([feature, zero], dim=2)# [bs, 256, 441]
        y = y.view(y.shape[0], y.shape[1], 21, 21) # [bs, 256, 21, 21]
        y = self.mlplayers(y)

        y = y.contiguous().view(y.shape[0], y.shape[1], -1)

        y = y.view(y.shape[0], -1)
        return y
    
if __name__ == '__main__':
    input=torch.randn(48,256,431).to(torch.device('cuda'))

    se1 = Reducedim1(channel=431).to(torch.device('cuda'))
    output=se1(input)
    print(output.shape)

    se2 = Reducedim2(channel=256).to(torch.device('cuda'))
    output=se2(input)
    print(output.shape)