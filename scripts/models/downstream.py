import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Union
from transformers.modeling_outputs import BaseModelOutput
import numpy as np
from .DilatedCNN import ConvNetConfig, ConvNetModel, OneHotEmbedding


class TransposeLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return x
    
        
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



class ConvNetForSupervised(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        vocab_size=7,
        n_layers=30,
        kernel_size=9,
        dilation_double_every=1,
        dilation_max=32,
        initializer_range = 0.02,
        dilation_cycle=6,
        output_size = 2,
        hidden_size_downstream = 64, 
        kernel_size_downstream=3, 
        upsample_factor : Union[bool, int] = False, 
        output_downsample_window = None,
        **kwargs, 
    ):
        
        super().__init__()
        self.config = ConvNetConfig(vocab_size=vocab_size, 
                                               hidden_size=hidden_size,
                                               n_layers=n_layers,
                                               kernel_size=kernel_size,
                                               dilation_double_every=dilation_double_every,
                                               dilation_max=dilation_max,
                                               dilation_cycle=dilation_cycle,
                                               initializer_range=initializer_range)
        

        self.encoder = ConvNetModel(self.config)

        self.downstream_cnn = CNN(input_size = hidden_size, output_size = output_size,
                                    hidden_size = hidden_size_downstream,
                                    kernel_size = kernel_size_downstream,
                                    upsample_factor = upsample_factor,
                                    output_downsample_window= output_downsample_window)

    def forward(self, x, activation = 'none', **kwargs):
        x = self.encoder(input_ids=x, **kwargs).last_hidden_state
        x = self.downstream_cnn(x, activation = activation)
        return x


       