'''
downstream.py
====================================
This module contains the implementations of the supervised models used in the paper.

- :class:`~bend.models.downstream.ConvNetForSupervised`: a ResNet that we train as baseline model on one-hot encodings, if no dedicated baseline architecture is available for a task.
- :class:`~bend.models.downstream.CNN`: a two-layer CNN used for all downstream tasks.
'''
from typing import Union
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from bend.models.dilated_cnn import ConvNetConfig, ConvNetModel, OneHotEmbedding

class CustomDataParallel(torch.nn.DataParallel):
    """
    A custom DataParallel class that allows for attribute access to the
    wrapped module.
    """
    def __getattr__(self, name):
        """Forward attribute access to the module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
        
class TransposeLayer(nn.Module):
    """A layer that transposes the input."""
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        """
        Transpose the input.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed tensor.
        """
        x = torch.transpose(x, 1, 2)
        return x
    
   
class UpsampleLayer(nn.Module):
    """
    A layer that upsamples the input along the sequence dimension.
    This is useful when a position in the input sequence corresponds to
    multiple positions in the output sequence. The one-to-n mapping
    needs to be a fixed factor.
    """

    def __init__(self, scale_factor=6, input_size = 2560):
        """
        Build an upsampling layer.
        
        Parameters
        ----------
        scale_factor: int
            The factor by which to upsample the input.

        input_size: int
            The embedding size of the input sequence.
        """
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.input_size = input_size
        
        self.upsample = nn.Sequential(TransposeLayer(), 
                                     nn.Upsample(scale_factor = scale_factor, 
                                                 mode = 'linear', 
                                                 align_corners = False), 
                                     TransposeLayer())

    def forward(self, x):
        """
        Upsample the input.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, embedding_size).

        Returns
        -------
        torch.Tensor
            Upsampled tensor. Has shape (batch_size, length * scale_factor, embedding_size).
        """
        x = self.upsample(x)
        return x #torch.reshape(x, (x.shape[0], -1, self.input_size))
    
class CNN(nn.Module):
    """
    A two-layer CNN with step size 1, ReLU activation, and a linear layer.
    """
    def __init__(self, input_size = 5, output_size = 2, 
                 hidden_size = 64, 
                 kernel_size=3, 
                 upsample_factor : Union[bool, int] = False, 
                 output_downsample_window = None,
                 encoder = None,
                 *args, **kwargs):
        """
        Build a two-layer CNN with step size 1, ReLU activation, and a linear layer.

        Parameters
        ----------
        input_size: int
            The embedding size of the input sequence.
        output_size: int
            The size of the output sequence.
        hidden_size: int
            The embedding size of the hidden layer.
        kernel_size: int
            The kernel size of the convolutional layers.
        upsample_factor: int
            The factor by which to upsample the input.
        output_downsample_window: int
            The window size for downsampling the output along the sequence dimension.
            This is done by taking the average of the output values in the window.
        """
        super(CNN, self).__init__()
        self.encoder = encoder 
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
                                        TransposeLayer(), 
                                        ) if output_downsample_window is not None else None
        self.linear = nn.Sequential(nn.Linear(hidden_size, np.prod(output_size) if isinstance(output_size, tuple) else output_size))
        self.softmax =  nn.Softmax(dim = -1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, activation = 'none', length = None, **kwargs):
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, embedding_size).
        activation: str
            The activation function to use. Can be 'softmax', 'softplus', 'sigmoid', or 'none'.
        length: int
            The actual length (in nucleotides) of the input sequence. Only required when embedding upsampling is used.
        Returns
        -------
        torch.Tensor
            Output tensor. Has shape (batch_size, output_length, output_size).
            output_length is determined by the input length, the upsampling factor, and the output downsampling window.

        """
        x = self.onehot_embedding(x)
        if hasattr(self, 'upsample'):
            x = self.upsample(x)[:, :length]
        if self.encoder is not None:
            x = self.encoder(input_ids=x, **kwargs).last_hidden_state
        # 1st conv layer
        x = self.conv1(x)
        # 2nd conv layer 
        x = self.conv2(x)
        if self.downsample is not None:
            x = self.downsample(x)
        # linear layer 
        x = self.linear(x)
        # reshape output if necessary
        if self.output_size == 1 and x.dim() > 2 or self.downsample:
            x = torch.flatten(x, 1)
        #    x = torch.reshape(x, (x.shape[0], x.shape[1], *self.output_size))
        # softmax
        if activation =='softmax': 
            x = self.softmax(x)
        elif activation == 'softplus':
            x = self.softplus(x)
        elif activation == 'sigmoid':
            x = self.sigmoid(x)
        return x



class ConvNetForSupervised(nn.Module):
    """
    A convolutional neural network for supervised learning.
    We use this as a baseline, when no dedicated supervised model
    for a particular task is available.
    """
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
        """
        Build a convolutional neural network for supervised learning.

        Parameters
        ----------
        hidden_size: int
            The size of the hidden layers.
        vocab_size: int
            The size of the input embeddings. This is called  `vocab_size` because in the one-hot encoding 
            case, the embedding size will be equal to the size of the vocabulary.
        n_layers: int
            The number of convolutional layers.
        kernel_size: int
            The kernel size of the convolutional layers.
        dilation_double_every: int
            The number of layers after which to double the dilation rate.
        dilation_max: int
            The maximum dilation rate.
        dilation_cycle: int
            The number of layers after which to reset the dilation rate to 1.
        output_size: int
            The size of the output sequence.
        hidden_size_downstream: int
            The embedding size of the hidden layer in the downstream CNN.
        kernel_size_downstream: int
            The kernel size of the convolutional layers in the downstream CNN.
        upsample_factor: int
            The factor by which to upsample the input.
        output_downsample_window: int
            The window size for downsampling the output along the sequence dimension.
            This is done by taking the average of the output values in the window.
        """

        
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
        self.softmax =  nn.Softmax(dim = -1)

    def forward(self, x, activation = 'none', **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, vocab_size).
        activation: str
            The activation function to use. Can be 'softmax', 'softplus', 'sigmoid', or 'none'.
        Returns
        -------
        torch.Tensor
            Output tensor. Has shape (batch_size, output_length, output_size).
            output_length is determined by the input length, the upsampling factor, and the output downsampling window.

        """
        x = self.encoder(input_ids=x, **kwargs).last_hidden_state
        x = self.downstream_cnn(x, activation = activation)

        return x

    
       