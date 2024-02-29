import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import Tensor
from torch.nn import init
import spconv.pytorch as spconv
import collections
import math
from core.cfgs import cfg
from torch.nn.parameter import Parameter


class MultiLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['n_head', 'in_features', 'out_features']
    n_head: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, n_head: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.empty((n_head, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(n_head, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        out = torch.einsum('kij, bkj -> bki', self.weight, input)
        if self.bias is not None:
            out += self.bias
        return out.contiguous()

    def extra_repr(self) -> str:
        return 'n_head={}, in_features={}, out_features={}, bias={}'.format(
            self.n_head, self.in_features, self.out_features, self.bias is not None
        )


class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).to(torch.device('cuda', cfg.GPU)))# .cuda()) # currently not compatible with running on CPU
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # Decoder is the base class for all decoders

    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class UpConv(Decoder):
    # UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, in_channels):
        super(UpConv, self).__init__()
        self.layer1 = self.upconv_module(in_channels)
        self.layer2 = self.upconv_module(in_channels//2)
        self.layer3 = self.upconv_module(in_channels//4)

class FCRNDecoder(nn.Module):
    def __init__(self):

        super(FCRNDecoder, self).__init__()
        # self.output_size = 56
        num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.decoder = UpConv(num_channels//2)

        # setting bias=true doesn't improve accuracy
        self.conv3 = nn.Conv2d(num_channels//16,1,kernel_size=3,stride=1,padding=1,bias=False)
        # self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        self.conv4 = nn.Conv2d(num_channels//16,1,kernel_size=3,stride=1,padding=1,bias=False)

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)

    def forward(self, x):
        return_dict = {}
        x = self.conv2(x)
        x = self.bn2(x)

        # decoder
        x = self.decoder(x)

        x1 = self.conv3(x)
        # x1 = self.bilinear(x1)
        return_dict['predict_depth'] = x1.squeeze(1)

        x2 = self.conv4(x)
        return_dict['predict_bc'] = x2.squeeze(1)

        return return_dict

class DEPTH_predict_layer(nn.Module):
    def upconv_module(self, in_channels):
        # UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
        upconv = nn.Sequential(collections.OrderedDict([
          ('unpool',    Unpool(in_channels)),
          ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
          ('batchnorm', nn.BatchNorm2d(in_channels//2)),
          ('relu',      nn.ReLU()),
        ]))
        return upconv

    def __init__(self, feat_dim=128):
        super(DEPTH_predict_layer, self).__init__()
        
        # self.drop = nn.Dropout2d()
        # self.conv3 = nn.Conv2d(feat_dim//2,1,kernel_size=3,stride=1,padding=1)
        # self.conv4 = nn.Conv2d(feat_dim//2,1,kernel_size=3,stride=1,padding=1)

        self.conv3 = nn.Conv2d(feat_dim,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv4 = nn.Conv2d(feat_dim,1,kernel_size=3,stride=1,padding=1,bias=False)
    
    def forward(self, x):
        return_dict = {}
        # x = self.drop(x)
        x1 = self.conv3(x)
        # x1 = self.bilinear(x1)
        return_dict['predict_depth'] = x1.squeeze(1)

        x2 = self.conv4(x)
        return_dict['predict_bc'] = x2.squeeze(1)

        return return_dict
    
class reduce_dim_layer(nn.Module):
    def __init__(self):
        super(reduce_dim_layer, self).__init__()
        channel = 67*4 # 268
        # self.reduconv1 = nn.Sequential(
        #     nn.Conv2d(768, channel, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(channel),
        #     nn.ReLU(inplace=True))
        # self.reduconv2 = nn.Sequential(
        #     nn.Conv2d(512, channel, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(channel),
        #     nn.ReLU(inplace=True))
        self.reduconv3 = nn.Sequential(
            nn.Conv2d(384, channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))
        
        # self.reduconv4 = nn.Sequential(
        #     nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(channel))
        # self.reduconv5 = nn.Sequential(
        #     nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(channel))
        # self.reduconv6 = nn.Sequential(
        #     nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(channel))
        self.reduconv6 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, x, i):
        # if i==0:
        #     s_feat = self.reduconv1(x)
        #     s_feat = self.reduconv4(s_feat)
        # if i==1:
        #     s_feat = self.reduconv2(x)
        #     s_feat = self.reduconv5(s_feat)
        if i==2:
            s_feat = self.reduconv3(x)
            s_feat = self.reduconv6(s_feat)
        
        return s_feat

class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = one_conv(256+8, 264, 'subm0')
        self.down0 = stride_conv(264, 512, 'down0')

        self.conv1 = one_conv(512, 512, 'subm2')
        self.down1 = stride_conv(512, 1024, 'down2')

        self.conv2 = one_conv(1024, 1024, 'subm3')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net = self.down1(net)

        net = self.conv2(net)
        net3 = net.dense()
        
        return net3


def one_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), 
        nn.ReLU())

# 测试网络模型
if __name__ == '__main__':
    batch_size = 1
    net = FCRNDecoder().cuda()
    x = torch.zeros(batch_size, 2048, 7, 7).cuda()
    print(net(x).size())











