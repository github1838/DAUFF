# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
# import sys
# sys.path.append(".")
from utils.myutils import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=51, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        # [64, 256, 512, 1024, 2048]
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                disp_pred = self.softmax(self.convs[("dispconv", i)](x))

                depth_range = torch.arange(self.num_output_channels).float().to(disp_pred.device)
                depth_range = depth_range.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                disp_idx_pred = torch.sum(disp_pred * depth_range, dim=1)

                self.outputs[("disp", i)] = disp_pred
                self.outputs[("disp_idx", i)] = disp_idx_pred

        return self.outputs
    
if __name__ == '__main__':
    depthDecoder = DepthDecoder(num_ch_enc=np.array([64, 256, 512, 1024, 2048]))
    print(depthDecoder)