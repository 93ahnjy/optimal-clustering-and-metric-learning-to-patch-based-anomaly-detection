import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math






def minmax_standardization(input_data):
    min_, max_ = input_data.min(), input_data.max()
    return (input_data - min_).div(max_ - min_ + 1e-8).data  # normalize


def mean_normalization(input_data):
    mean_, std_ = input_data.mean(), input_data.std()
    return (input_data - mean_) / std_





class Print_layer(nn.Module):
    def __init__(self, string=None):
        super(Print_layer, self).__init__()
        self.string = string

    def forward(self, x):
        print(x.size())
        if self.string is not None:
            print(self.string)
        return x




class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)




class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)





class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


