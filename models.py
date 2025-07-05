# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
from torch.nn.utils import weight_norm
# from tornado.escape import squeeze

import MABN
# from torch_geometric.nn import ChebConv

from RevIN.RevIN import RevIN
from attention.Axial_attention import SelfAttention
from attention.CBAM import CBAMBlock
from attention.SEAttention import SEAttention
from attention.SelfAttention import ScaledDotProductAttention
# from mamba_ssm import Mamba, Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2_simple import Mamba2Simple
# from GCFC import CfC
# from ncps.torch import CfC
# from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from sklearn.decomposition import PCA

# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1, dropout=0.25):
#         super(SimpleRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.rnn = Mamba(
#             d_model=32, # Model dimension d_model
#             d_state=16,  # SSM state expansion factor
#             d_conv=4,  # Local convolution width
#             expand=2,  # Block expansion factor
#         )
#
#         self.linear = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x, hidden):
#         #output, hidden = self.rnn(x, hidden) if isinstance(self.rnn(x, hidden), tuple) else (self.rnn(x, hidden), None)
#
#         output, hidden = self.rnn(x, hidden)
#
#         pred = self.linear(output[:, -1, :])
#         return pred, hidden
#
#     def init_hidden(self):
#         return torch.randn(1, 24, self.hidden_size)
#
#
#
# class SimpleGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, in_channels,output_size, num_layers):
#         super(SimpleGRU, self).__init__()
#         self.hidden_size = hidden_size
# #####
#         self.cbam = CBAMBlock(channel=24, reduction=24, kernel_size=1)
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
#             torch.nn.BatchNorm2d(64),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
#             torch.nn.BatchNorm2d(1),
#             torch.nn.ReLU()
#         )
#         # self.mlp = torch.nn.Sequential(
#         #     nn.Linear(216, 128),
#         #     torch.nn.ReLU(),
#         #     nn.Linear(128, 64),
#         #     torch.nn.ReLU(),
#         #     nn.Linear(64, 32)
#         # )
#         # self.Mamba = Mamba(d_model=12, d_state=64, d_conv=3, expand=5).to("cuda")
#         self.gru = nn.GRU(input_size=12, hidden_size=32, num_layers=2, dropout=0.25, batch_first=True).to("cuda")
#         # self.Mamba = Mamba(d_model=12, d_state=d_state, d_conv=d_conv, expand=expand).to("cuda")  #PSO
#         # self.elu = nn.ELU()
#         self.linear = nn.Linear(32, 1).to("cuda")
#     def forward(self, x, hidden):
#         x = self.cbam(x.transpose(1, 2)).transpose(1, 2)
#         conv_out = self.conv(x).squeeze(0).to("cuda") # Mamba
#         # output = self.mlp(x).to("cuda")
#         output, hidden = self.gru(conv_out, hidden)
#         # output1 = self.Mamba(output) # Mamba
#         # # combined_output = torch.cat((output, y), dim=2)
#         # output2, hidden = self.gru2(output1, hidden)
#         pred = self.linear(output[:, -1, :]).to("cpu")
#         return pred, hidden
#
#     def init_hidden(self):
#         return torch.randn(1, 24, self.hidden_size)
#
#
# class SimpleLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
#         super(SimpleLSTM, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=17, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
#             torch.nn.BatchNorm2d(64),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
#             torch.nn.BatchNorm2d(1),
#             torch.nn.ReLU()
#         )
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             # dropout=dropout,
#             batch_first=True
#         ).to("cuda")
#         self.Mamba = Mamba(
#             # This module uses roughly 3 * expand * d_model^2 parameters
#             d_model=12,  # Model dimension d_model
#             d_state=4,  # SSM state expansion factor
#             d_conv=4,  # Local convolution width
#             expand=4,  # Block expansion factor
#         ).to("cuda")
#         self.linear = nn.Linear(44, output_size).to("cuda")
#         # self.linear = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = self.conv(x).squeeze(0).to("cuda")  # Mamba
#         output, (h_n, c_n) = self.lstm(x)
#         conv_out = self.Mamba(x)  # Mamba
#         # conv_out = conv_out + output
#         combined_output = torch.cat((conv_out, output), dim=2)
#         pred = self.linear(combined_output[:, -1, :]).to("cpu")
#         return pred
#
#     def init_hidden(self):
#         return torch.randn(1, 24, self.hidden_size)

##############################################dimension
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity='relu',    # 'tanh' or 'relu'
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        pred = self.linear(output[:, -1, :])
        return pred, hidden

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x = x.mean(dim=1)  # 移除第一个维度
        # x = x.squeeze(dim=1)
        output, hidden = self.gru(x, hidden)
        # output=self.dropout(output)
        pred = self.linear(output[:, -1, :])
        return pred, hidden

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.25):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            #dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x[:, 11, :, :]  #####1013
        output, (h_n, c_n) = self.lstm(x)
        pred = self.linear(output[:, -1, :])
        return pred

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)





##################################################
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)     #
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x=x[:,11,:,:]  #####1013
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred

class SimpleMamba(nn.Module):
    def __init__(self,d_model,d_state, d_conv, expand, d_inner, dt_rank):
        super(SimpleMamba, self).__init__()
        self.Mamba = Mamba(d_model=12, d_state=d_state, d_conv=d_conv, expand=expand, d_inner=d_inner, dt_rank=dt_rank).to('cuda')#
        self.linear = nn.Linear(12, 1)

    def forward(self, x):

        x=x.to('cuda')
        output=self.Mamba(x).to('cpu')
        pred = self.linear(output[:, -1, :])
        return pred

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):   #0.2
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

  ######################
        # self.conv3 = weight_norm(nn.Conv1d(n_outputs, n_outputs, 2,
        #                                    stride=stride, padding=padding, dilation=dilation))
        # self.chomp3 = Chomp1d(padding)
        # self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(dropout)
    ####################3
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
                                 # self.conv3, self.chomp3, self.relu3, self.dropout3)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # self.conv3.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
        # return self.relu(out)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):  #0.2
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            # layers += [TemporalBlock(in_channels, out_channels, int(kernel_size / (i + 1)), stride=1, dilation=dilation_size,
            #                          padding=(kernel_size - 1) * dilation_size, dropout=dropout)]  ##########gai -i
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class STCN(nn.Module):
    def __init__(self, input_size, in_channels, output_size, num_channels, kernel_size, dropout):
        super(STCN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )
        # self.Mamba = Mamba(d_model=216, d_state=16, d_conv=2, expand=1)  # PSO
        # self.linear1 = nn.Linear(12, 32)
        # self.att = ScaledDotProductAttention(d_model=32, d_k=32, d_v=32, h=2)
        # self.Mamba = Mamba(d_model=32, d_state=16, d_conv=2, expand=1)  # PSO
        # self.cbam = CBAMBlock(channel=24, reduction=12, kernel_size=4)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        # self.linear1 = nn.Linear(12, 32)
        # self.Mamba = Mamba(d_model=32, d_state=8, d_conv=4, expand=2)  # PSO
        # self.Mamba = Mamba(d_model=216, d_state=d_state, d_conv=d_conv, expand=expand, d_inner=d_inner, dt_rank=dt_rank).to("cuda")#PSO
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x = self.cbam(x.transpose(1, 2)).transpose(1, 2)
        conv_out = self.conv(x).squeeze(1)
        # conv_out1 = self.linear1(conv_out)
        # output1 = self.Mamba(conv_out1)
        # x = self.Mamba(x)
        # output2 = self.att(conv_out1, conv_out1, conv_out1)
        output = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)
        # output2 = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)
        # output = self.att(output,output,output)
        # output = output1 + output2
        # output = self.Mamba(output)

        pred = self.linear(output[:, -1, :])
        return pred


class oMamba(nn.Module):
    def __init__(self, input_size, in_channels, out_in_channels, d_state, d_conv, expand, d_inner, dt_rank):
        super(oMamba, self).__init__()
        # self.mabn = MABN
        # self.se = SEAttention(channel=18, reduction=18)
        self.conv = torch.nn.Sequential(
            # MABN.CenConv2d(in_planes=in_channels, out_planes=out_in_channels, kernel_size=1),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_in_channels, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(out_in_channels),
            # MABN.MABN2d(out_in_channels,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_in_channels, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            # MABN.CenConv2d(in_planes=out_in_channels, out_planes=1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            # MABN.MABN2d(1,1,1),
            torch.nn.ReLU()
        )
        #
        # # self.Mamba = Mamba(d_model=216, d_state=4, d_conv=2, expand=2).to("cuda")
        # # self.mlp = torch.nn.Sequential(
        # #     nn.Linear(216,64),
        # #     torch.nn.ReLU(),
        # #     nn.Linear(64, 32),
        # #     torch.nn.ReLU()
        # # ).to("cuda")
        # # self.conv1d = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=1).to("cuda")
        # # self.rnn = CfC(12, 64).to("cuda")
        self.Mamba = Mamba(d_model=12, d_state=d_state, d_conv=d_conv, expand=expand, d_inner=d_inner, dt_rank=dt_rank).to('cuda')# PSO
        # self.Mamba = Mamba2Simple(d_model=12,
        #                         d_state=64,  # SSM state expansion factor, typically 64 or 128
        #                         d_conv=4,  # Local convolution width
        #                         expand=2,  # Block expansion factor
        #                         headdim=1 # default 64
        # ).to('cuda')
        # self.cfc = CfC(12, 32, mixed_memory=True)
        # self.downsample = nn.Conv1d(12, 12, 1)
        # self.relu = nn.ReLU()
        self.linear = nn.Linear(12, 1)


    def forward(self, x):
        # x = self.se(x)
        conv_out = self.conv(x).squeeze(0).to('cuda') # Mamba
        # x = conv_out.permute(0,2,1)
        # x = self.rnn(conv_out)
        # x = x.permute(0, 2, 1)
        # out = self.Mamba(conv_out)
        # res = self.downsample(conv_out.transpose(1, 2)).transpose(1, 2)
        # res = self.Mamba(res)
        # output = out + res
        output = self.Mamba(conv_out).to('cpu')
        # output = self.relu(out + res)
        # output = self.mlp(output)
        # output2 = self.atten(conv_out, conv_out, conv_out)
        # output = self.Mamba(conv_out.to("cuda"))
        # output = self.cfc(conv_out)
        # output = output[0]
        pred = self.linear(output[:, -1, :])
        return pred

class gMamba(nn.Module):
    def __init__(self, input_size,  in_channels, out_in_channels, d_state, d_conv, expand, d_inner, dt_rank):
        super(gMamba, self).__init__()
#####
        # self.cbam = CBAMBlock(channel=24, reduction=24, kernel_size=1)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_in_channels, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(out_in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_in_channels, out_channels=1, kernel_size=(1, 1), stride=1, padding=0),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU()
        )
        # self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.Mamba = Mamba(d_model=12, d_state=d_state, d_conv=d_conv, expand=expand, d_inner=d_inner, dt_rank=dt_rank)#PSO
        self.gru = nn.GRU(input_size=12, hidden_size=6, dropout=0.25, batch_first=True, bidirectional=True)
        # self.Mamba = Mamba(d_model=12, d_state=1, d_conv=2, expand=5).to("cuda")
        # self.Mamba = Mamba(d_model=12, d_state=d_state, d_conv=d_conv, expand=expand).to("cuda")
        # self.elu = nn.ELU()
        self.linear = nn.Linear(12, 1)

    def forward(self, x, hidden):
        # x = self.cbam(x.transpose(1, 2)).transpose(1, 2)
        conv_out = self.conv(x).squeeze(0) # Mamba
        # output = self.cfc(x)
        output1 = self.Mamba(conv_out)
        # output1 = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)
        # output1 = self.tcn(conv_out.transpose(1, 2)).transpose(1, 2)
        output2, hidden = self.gru(conv_out, hidden)
        output = output1 + output2

        pred = self.linear(output[:, -1, :]).to("cpu")
        return pred, hidden

    def init_hidden(self):
        return torch.randn(1, 24, self.hidden_size)