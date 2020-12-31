import torch
import torch.nn as nn
import numpy as np
from torchvision.models import wide_resnet50_2
from torchsummary import summary
from layers import*
from cbam import CBAM



# prerequisites
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, cv2
import math
import torchvision
import random
import pickle

from layers import Identity, Flatten
from functions import*
from evaluate import*
from evaluate_Kmeans import*
from layers import*
from Dataset import*

from functools import reduce



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def load_logs(class_idx):
    if os.path.exists(f'models/result_log_{class_idx}.pkl'):
        with open(f'models/result_log_{class_idx}.pkl', 'rb') as f:
            result_log = pickle.load(f)
        print(" **** Loads previous params.", result_log)




def forward_hier(x, emb_small, K):
    K_2 = K // 2
    n = x.size(0)
    x1 = x[..., :K_2, :K_2]
    x2 = x[..., :K_2, K_2:]
    x3 = x[..., K_2:, :K_2]
    x4 = x[..., K_2:, K_2:]
    xx = torch.cat([x1, x2, x3, x4], dim=0)
    hh = emb_small(xx)

    h1 = hh[:n]
    h2 = hh[n: 2 * n]
    h3 = hh[2 * n: 3 * n]
    h4 = hh[3 * n:]

    h12 = torch.cat([h1, h2], dim=3)
    h34 = torch.cat([h3, h4], dim=3)
    h = torch.cat([h12, h34], dim=2)
    return h







class EncoderDeep(nn.Module):
    def __init__(self, class_idx, K, D=64, bias=True):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0, bias=bias),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, 3, 1, 0, bias=bias),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 3, 1, 0, bias=bias),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 3, 1, 0, bias=bias),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, 3, 1, 0, bias=bias),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 32, 3, 1, 0, bias=bias),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 32, 3, 1, 0, bias=bias),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, D, 3, 1, 0, bias=bias),
            nn.BatchNorm2d(D),
            nn.Tanh(),
        )

        self.K = K
        self.D = D
        self.class_idx = class_idx

    def forward(self, x):
        h = self.model(x)
        return h

    def save(self):
        torch.save(self.state_dict(), f'./models/Patch_SVDD_EncoderHier_small_{self.class_idx}_{self.K}.pth')

    def load(self):
        self.load_state_dict(torch.load(f'./models/Patch_SVDD_EncoderHier_small_{self.class_idx}_{self.K}.pth'))






class EncoderHier(nn.Module):
    def __init__(self, class_idx, K, D=64, bias=True):
        super().__init__()

        self.enc = EncoderDeep(K // 2, D, bias=bias)

        self.enc_big = nn.Sequential(
            nn.Conv2d(D, 128, 2, 1, 0, bias=bias),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, D, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(D),
            nn.Tanh(),
        )

        self.K = K
        self.D = D
        self.class_idx = class_idx

    def forward(self, x):
        h = forward_hier(x, self.enc, K=self.K)
        h = self.enc_big(h)
        return h

    def save(self):
        torch.save(self.state_dict(), f'./models/Patch_SVDD_EncoderHier_big_{self.class_idx}_{self.K}.pth')

    def load(self):
        self.load_state_dict(torch.load(f'./models/Patch_SVDD_EncoderHier_big_{self.class_idx}_{self.K}.pth'))








class PositionClassifier(nn.Module):

    def __init__(self, class_idx, K, D=64, class_num=8):               # K는 모델 이름 짓는데만 사용된다.
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(D, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            NormalizedLinear(128, class_num),
            #nn.Linear(128, class_num)
        )

        self.D = D
        self.K = K
        self.class_idx = class_idx


    def forward(self, h1, h2):
        h1 = h1.view(-1, self.D)
        h2 = h2.view(-1, self.D)
        return self.model(h1 - h2)


    def save(self):
        torch.save(self.model.state_dict(), f'./models/Patch_SVDD_PositionClassifier_{self.class_idx}_{self.K}.pth')

    def load(self):
        self.model.load_state_dict(torch.load(f'./models/Patch_SVDD_PositionClassifier_{self.class_idx}_{self.K}.pth'))



# class GC_block(nn.Module):
#     def __init__(self, in_channel, reduction_ratio=8, bias=True):
#         super().__init__()
#         self.conv2d_1 = nn.Conv2d(in_channel, 1, 1, 1, 0, bias=bias)
#         self.conv2d_2 = nn.Conv2d(in_channel, in_channel // reduction_ratio, 1, 1, 0, bias=bias)
#         self.ln  = nn.LayerNorm(in_channel // reduction_ratio)
#         self.act = nn.LeakyReLU(0.1)
#         self.conv2d_3 = nn.Conv2d(in_channel // reduction_ratio, in_channel, 1, 1, 0, bias=bias)
#
#         self.reduction_ratio = reduction_ratio
#
#
#     def forward(self, x):
#
#         B, C, H, W = x.size()
#
#
#
#         x1 = self.conv2d_1(x)               # B, 1, H, W
#         x1 = x1.view(-1, H*W, 1)            # B, H*W, 1
#         x1 = torch.softmax(x1, dim=1)
#
#         x2 = x.view(-1, C, H*W)              # B, C, H*W
#
#         x3 = torch.bmm(x2, x1)
#         x3 = x3.view(-1, C, 1, 1)
#         x3 = self.conv2d_2(x3)
#
#         x3 = x3.view(-1, C // self.reduction_ratio)
#         x3 = self.ln(x3)
#
#         x3 = x3.view(-1, C // self.reduction_ratio, 1, 1)
#         x3 = self.conv2d_3(x3)
#
#         x = x + x3
#
#         return x
#
#
#
#
#
#
#
# class Basic_block(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, stride, padding, activation=nn.LeakyReLU(0.1), bias=True, use_cbam=False):
#         super().__init__()
#         self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
#         self.bn = nn.BatchNorm2d(out_channel)
#         self.activation = activation
#
#         self.use_cbam = use_cbam
#         if use_cbam:
#             self.cbam = CBAM(out_channel)
#             #self.cbam = GC_block(out_channel)
#
#
#
#     def forward(self, x):
#         x = self.conv2d(x)
#         x = self.bn(x)
#
#         if self.use_cbam:
#             att = self.cbam(x)
#             x = x + att
#         h = self.activation(x)
#
#         return h
#
#
#
#
#
# class Basic_block_FC(nn.Module):
#     def __init__(self, in_channel, out_channel, activation=nn.LeakyReLU(0.1)):
#         super().__init__()
#
#         self.fc = nn.Linear(in_channel, out_channel)
#         self.bn = nn.BatchNorm1d(out_channel)
#         self.activation = activation
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.bn(x)
#         h = self.activation(x)
#         return h
#
#
#
#
#
#
#
#
#
# def forward_hier(x, emb_small, K):
#
#     """ Don't use 64 x 64 size input """
#
#     ''' Reshape batch input (B, 3, 64, 64) to  (4*B, 3, 32, 32) '''
#     ''' output h = (B, 64, 2, 2) '''
#
#     K_2 = K // 2
#     n = x.size(0)
#     x1 = x[..., :K_2, :K_2]
#     x2 = x[..., :K_2, K_2:]
#     x3 = x[..., K_2:, :K_2]
#     x4 = x[..., K_2:, K_2:]
#     xx = torch.cat([x1, x2, x3, x4], dim=0)
#     hh = emb_small(xx)
#
#     h1 = hh[:n]
#     h2 = hh[n: 2 * n]
#     h3 = hh[2 * n: 3 * n]
#     h4 = hh[3 * n:]
#
#     h12 = torch.cat([h1, h2], dim=3)
#     h34 = torch.cat([h3, h4], dim=3)
#     h = torch.cat([h12, h34], dim=2)
#     return h
#
#
#
#
#
#
#
#
#
#
#
#
# class EncoderDeep(nn.Module):
#     def __init__(self, class_idx, K, D=64, bias=True):
#         super().__init__()
#
#         self.block1 = Basic_block(3, 32, 3, 2, 0, bias=bias)
#         self.block2 = Basic_block(32, 64, 3, 1, 0, bias=bias)
#         self.block3 = Basic_block(64, 128, 3, 1, 0, bias=bias)
#         self.block4 = Basic_block(128, 128, 3, 1, 0, bias=bias)
#         self.block5 = Basic_block(128, 64, 3, 1, 0, bias=bias)
#         self.block6 = Basic_block(64, 32, 3, 1, 0, bias=bias)
#         self.block7 = Basic_block(32, 32, 3, 1, 0, bias=bias)
#         self.last_block = Basic_block(32, D, 3, 1, 0, bias=bias, activation=nn.Tanh(), use_cbam=False)
#
#         self.K = K
#         self.D = D
#         self.class_idx = class_idx
#         self.activations = dict()
#         self.features = []
#
#
#     def forward(self, x):
#         self.features = []
#
#         h = self.block1(x)                  # [-1, 32, 15, 15]
#         h = self.block2(h)                  # [-1, 64, 13, 13]
#         h = self.block3(h)                  # [-1, 128, 11, 11]
#         h = self.block4(h)                  # [-1, 128, 9, 9]
#         h = self.block5(h)                 # [-1, 64, 7, 7]
#         h = self.block6(h)                 # [-1, 32, 5, 5]
#         h = self.block7(h)                  # [-1, 32, 3, 3]
#         h = self.last_block(h)             # [-1, D, 1, 1]
#
#         #self.features = [h2, h5]
#
#         return h
#
#     def save(self):
#         torch.save(self.state_dict(), f'./models/Patch_SVDD_EncoderHier_small_{self.class_idx}_{self.K}.pth')
#
#     def load(self):
#         self.load_state_dict(torch.load(f'./models/Patch_SVDD_EncoderHier_small_{self.class_idx}_{self.K}.pth'))





















# class EncoderHier(nn.Module):
#     def __init__(self, class_idx, K, D=64, bias=True):
#         super().__init__()
#
#         self.enc = EncoderDeep(K // 2, D, bias=bias)
#
#         self.enc_big = nn.Sequential(
#             Basic_block(D, 128, 2, 1, 0, bias=bias),
#             Basic_block(128, D, 1, 1, 0, bias=bias, activation=nn.Tanh()),
#         )
#
#         self.K = K
#         self.D = D
#         self.class_idx = class_idx
#         self.features = []
#
#     def forward(self, x):
#         self.features = []
#
#         hh = forward_hier(x, self.enc, K=self.K)
#         h = self.enc_big(hh)
#         #self.features = [hh, h]
#
#         return h
#
#     def save(self):
#         torch.save(self.state_dict(), f'./models/Patch_SVDD_EncoderHier_big_{self.class_idx}_{self.K}.pth')
#
#     def load(self):
#         self.load_state_dict(torch.load(f'./models/Patch_SVDD_EncoderHier_big_{self.class_idx}_{self.K}.pth'))
#
#
#
#
#
#
#
#
# class PositionClassifier(nn.Module):
#
#     def __init__(self, class_idx, K, D=64, class_num=8):               # K는 모델 이름 짓는데만 사용된다.
#         super().__init__()
#
#         self.model = nn.Sequential(
#             Basic_block_FC(D, 128),
#             Basic_block_FC(128, 128),
#
#             NormalizedLinear(128, class_num),
#             #nn.Linear(128, class_num)
#         )
#
#         self.D = D
#         self.K = K
#         self.class_idx = class_idx
#
#
#     def forward(self, h1, h2):
#         h1 = h1.view(-1, self.D)
#         h2 = h2.view(-1, self.D)
#         return self.model(h1 - h2)
#
#
#     def save(self):
#         torch.save(self.model.state_dict(), f'./models/Patch_SVDD_PositionClassifier_{self.class_idx}_{self.K}.pth')
#
#     def load(self):
#         self.model.load_state_dict(torch.load(f'./models/Patch_SVDD_PositionClassifier_{self.class_idx}_{self.K}.pth'))











































































if __name__ == '__main__':

    # model = wide_resnet50_2(pretrained=True).cuda()
    # summary(model, input_size=(3, 224, 224))


    model1 = EncoderDeep(K=32, D=64, class_idx=1).cuda()
    model2 = EncoderHier(K=64, D=64, class_idx=1).cuda()

    summary(model1, input_size=(3, 32, 32))
    summary(model2, input_size=(3, 64, 64))

    test_input1 = torch.ones((56, 3, 32, 32)).cuda()
    test_input2 = torch.ones((56, 3, 64, 64)).cuda()

    output1 = model1(test_input1)
    output2 = model2(test_input2)

    h1_ = model1.features[0]
    h2_ = model2.features[0]

    h1_ = h1_.mean(2).mean(2)
    h2_ = h2_.mean(2).mean(2)

    print(h1_.shape, h2_.shape)



