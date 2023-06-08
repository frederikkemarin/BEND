'''
Supervised models to be trained on embedded or one-hot encoded sequences.
'''
from typing import Union
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


class TransposeLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x
    
class OneHotEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        if x.dim() > 2:
            return x
        else: # if categorically encoded 
            return F.one_hot(x.long(), num_classes=self.hidden_size).float()
        
class UpsampleLayer(nn.Module):

    def __init__(self, scale_factor=6, input_size = 2560):
        '''
        Upsample the length of a sequence by scale_factor.
        input dim (batch_size, length, embedding_size)
        output_dim (batch_size, length * scale_factor, embedding_size)
        '''
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.input_size = input_size
        
        self.upsample = nn.Sequential(TransposeLayer(), 
                                     nn.Upsample(scale_factor = scale_factor, 
                                                 mode = 'linear', 
                                                 align_corners = False), 
                                     TransposeLayer())
        '''
        self.upsample = nn.Sequential(TransposeLayer(), 
                                     nn.ConvTranspose1d(input_size, input_size*scale_factor, kernel_size = 1), 
                                     TransposeLayer())
        '''
    
    def forward(self, x):
        x = self.upsample(x)
        return x #torch.reshape(x, (x.shape[0], -1, self.input_size))
    
class CNN(nn.Module):
    '''
        Two layer CNN with step size 1, Relu activation,
        Finish with linear layer to output size per position
        output should be same length as input
    '''
    def __init__(self, input_size = 5, output_size = 2, 
                 hidden_size = 64, 
                 kernel_size=3, 
                 upsample_factor : Union[bool, int] = False, 
                 output_downsample_window = None,
                 *args, **kwargs):
        super(CNN, self).__init__()
        self.output_size = output_size
        self.onehot_embedding = OneHotEmbedding(input_size)
        if upsample_factor: 
            self.upsample = UpsampleLayer(scale_factor = upsample_factor)
        
        self.conv1 = nn.Sequential(TransposeLayer(), 
                                   nn.Conv1d(input_size, hidden_size, kernel_size, stride = 1, padding = 1), 
                                   TransposeLayer(),
                                   nn.GELU())
        
        self.conv2 = nn.Sequential(TransposeLayer(), 
                                   nn.Conv1d(hidden_size, hidden_size, kernel_size, stride = 1, padding = 1), 
                                   TransposeLayer(), 
                                   nn.GELU(),
                                  )

        self.downsample = nn.Sequential(TransposeLayer(), 
                                        nn.AvgPool1d(kernel_size = output_downsample_window, 
                                                     stride = output_downsample_window), 
                                        TransposeLayer(),) if output_downsample_window is not None else None
        self.linear = nn.Sequential(nn.Linear(hidden_size, np.prod(output_size) if isinstance(output_size, tuple) else output_size))
        self.softmax =  nn.Softmax(dim = -1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, activation = 'none', length = None):
        x = self.onehot_embedding(x)
        if hasattr(self, 'upsample'):
            x = self.upsample(x)[:, :length]
        # 1st conv layer
        x = self.conv1(x)
        # 2nd conv layer 
        x = self.conv2(x)
        if self.downsample is not None:
            x = self.downsample(x)
        # linear layer 
        x = self.linear(x)
        # reshape output if necessary
        if isinstance(self.output_size, tuple):
            x = torch.reshape(x, (x.shape[0], x.shape[1], *self.output_size))
        # softmax
        if activation =='softmax': 
            x = self.softmax(x)
        elif activation == 'softplus':
            x = self.softplus(x)
        elif activation == 'sigmoid':
            x = self.sigmoid(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        hidden_size=None,
        **kwargs,
    ):
        
        super().__init__()
        self.conv = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                padding="same",
                **kwargs,
            ),
            TransposeLayer(),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x



def get_dilation_schedule(dilation_max = 32, dilation_cycle = 6, dilation_double_every =1, n_layers=30):
    return [
        min(dilation_max, 2**((i%dilation_cycle)//dilation_double_every))
        for i in range(n_layers)
    ]


class DilatedConvNet(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        n_layers=30,
        kernel_size=9,
        dilation_double_every=1,
        dilation_max=32,
        dilation_cycle=6,
        output_size = 2,
        hidden_size_downstream = 64, 
        kernel_size_downstream=3, 
        upsample_factor : Union[bool, int] = False, 
        output_downsample_window = None,
        **kwargs, 
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.onehot_embedding = OneHotEmbedding(hidden_size)

        self.dilation_schedule = get_dilation_schedule(dilation_max, dilation_cycle, dilation_double_every, n_layers)
        self.encoder = nn.Sequential(*[
            ConvLayer(
                hidden_size=hidden_size,
                kernel_size=kernel_size,
                dilation=self.dilation_schedule[i],
            )
            for i in range(n_layers)
        ])
        

        self.downstream_cnn = CNN(input_size = hidden_size, output_size = output_size,
                                    hidden_size = hidden_size_downstream,
                                    kernel_size = kernel_size_downstream,
                                    upsample_factor = upsample_factor,
                                    output_downsample_window= output_downsample_window)


    def forward(self, x, activation = 'none', **kwargs):
        x = self.onehot_embedding(x)
        x = self.encoder(x)
        x = self.downstream_cnn(x, activation = activation)

        return x

    
       