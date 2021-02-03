import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LocalConv1d(nn.Module):
    def __init__(self, input_channels,  out_channels, output_size, kernel_size, stride=1, bias=False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.Tensor(1, out_channels, output_size, kernel_size * input_channels).to(device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(1, output_channels, output_size).to(device)
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.kernel_size = kernel_size
        self.stride = stride

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        kw = self.kernel_size
        dw = self.stride
        x = x.unfold(1, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([-1])
        if self.bias is not None:
            out += self.bias
            
        return out
        
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        return output