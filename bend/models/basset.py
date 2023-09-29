'''
basset.py
====================================
This module contains the implementation of the Basset model.

- :class:`~bend.models.downstream.Basset`: The Basset CNN model architecture.
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
    

    
class Basset(nn.Module):
    """
    The Basset model.
    """
    def __init__(self, input_size = 5, input_len = 512, output_size = 2, 
                 upsample_factor : Union[bool, int] = False, 
                 encoder = None,
                 *args, **kwargs):
        """
        Build the Basset model.


        Parameters
        ----------
        input_size: int
            The embedding size of the input sequence.
        input_len: int
            The length of the input sequence.
        output_size: int
            The size of the output sequence.
        upsample_factor: int
            The factor by which to upsample the input.
        """
        super(Basset, self).__init__()
        self.encoder = encoder 
        self.output_size = output_size
        self.onehot_embedding = OneHotEmbedding(input_size)
        if upsample_factor: 
            self.upsample = UpsampleLayer(scale_factor = upsample_factor)


        self.inp_tranpose = TransposeLayer()

        layers = []
        prev_input_size = input_size
        out_len = None
        for i, (kernels, kernel_size, pool) in enumerate([(300, 19, 3), (200, 11, 4), (200, 7, 4)]):
            layers.append(nn.Conv1d(prev_input_size, kernels, kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(kernels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool, pool))
            prev_input_size = kernels
            out_len = input_len//pool

        self.conv_net = nn.Sequential(*layers)

        # TODO the 10 is being implied by the seq len coming in.
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_input_size*out_len, 1000),
            nn.ReLU,
            nn.Dropout(0.3),
            nn.Linear(1000, 1000),
            nn.Dropout(0.3),
            nn.Linear(1000, output_size)

        )
        self.sigmoid = nn.Sigmoid()


        
    def forward(self, x, activation = 'none', length = None, **kwargs):
        """
        Forward pass of Basset.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, embedding_size).
        activation: str
            The activation function to use. Can be 'sigmoid', or 'none'.
        length: int
            The actual length (in nucleotides) of the input sequence. Only required when embedding upsampling is used.
        Returns
        -------
        torch.Tensor
            Output tensor. Has shape (batch_size, output_size).
            output_length is determined by the input length, the upsampling factor, and the output downsampling window.

        """
        x = self.onehot_embedding(x)
        if hasattr(self, 'upsample'):
            x = self.upsample(x)[:, :length]
        if self.encoder is not None:
            x = self.encoder(input_ids=x, **kwargs).last_hidden_state

        x = self.inp_tranpose(x)
        x = self.conv_net(x)
        x = self.clf(x)

        if activation == 'sigmoid':
            x = self.sigmoid(x)
        return x