"""
dilated_cnn.py
====================================
A ResNet with dilated convolutions masked language model.
code from https://github.com/songlab-cal/gpn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput


class ConvNetConfig(PretrainedConfig):
    """Configuration of a ResNet with dilated convolutions."""
    model_type = "ConvNet"

    def __init__(
        self,
        vocab_size=7,
        hidden_size=512,
        n_layers=30,
        kernel_size=9,
        dilation_double_every=1,
        dilation_max=32,
        dilation_cycle=6,
        initializer_range=0.02,
        **kwargs
    ):
        """
        Build the configuration of a ResNet with dilated convolutions.

        Parameters
        ----------
        vocab_size: int
            Size of the vocabulary.
        hidden_size: int
            Size of the hidden state.
        n_layers: int
            Number of layers.
        kernel_size: int
            Size of the kernel.
        dilation_double_every: int
            Number of layers after which the dilation is doubled.
        dilation_max: int
            Maximum dilation.
        dilation_cycle: int
            Number of layers after which the dilation is reset.
        initializer_range: float
            Range of the initializer.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_double_every = dilation_double_every
        self.dilation_max = dilation_max
        self.dilation_cycle = dilation_cycle
        self.initializer_range = initializer_range


class ConvNetPreTrainedModel(PreTrainedModel):
    """Base class for a ResNet with dilated convolutions.
    Hanndles the initialization, loading and saving of the model.
    """
    config_class = ConvNetConfig
    base_model_prefix = "model"
    #supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


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


class ConvLayer(nn.Module):
    """A layer that performs a convolution."""
    def __init__(
        self,
        hidden_size=None,
        **kwargs,
    ):
        """
        Build a convolutional layer.
        
        Parameters
        ----------
        hidden_size: int
            Size of the hidden state.
        **kwargs
            Additional arguments passed to nn.Conv1d.
        """
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
        """
        Perform a convolution.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class OneHotEmbedding(nn.Module):
    """A layer that performs a one-hot embedding."""
    def __init__(
        self,
        hidden_size=None,
    ):
        """
        Build a one-hot embedding layer.

        Parameters
        ----------
        hidden_size: int
            Size of the hidden state - this is equal to the size of the vocabulary.
        """
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Perform a one-hot embedding. If the input is already one-hot embedded 
        (it has two dimensions), then it is returned as is.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
        """
        if x.dim() > 2: # already onehot embedded 
            return x
        else: # if categorically encoded 
            return F.one_hot(x.long(), num_classes=self.hidden_size).float()



def _get_dilation_schedule(config):
    return [
        min(config.dilation_max, 2**((i%config.dilation_cycle)//config.dilation_double_every))
        for i in range(config.n_layers)
    ]


class ConvNetModel(ConvNetPreTrainedModel):
    """A ResNet with dilated convolutions."""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        """
        Build a ResNet with dilated convolutions.

        Parameters
        ----------
        config: ConvNetConfig
            Configuration for the model.
        """
        super().__init__(config)
        self.config = config

        self.embedding = OneHotEmbedding(config.hidden_size)

        self.dilation_schedule = _get_dilation_schedule(config)
        self.encoder = nn.Sequential(*[
            ConvLayer(
                hidden_size=config.hidden_size,
                kernel_size=config.kernel_size,
                dilation=self.dilation_schedule[i],
            )
            for i in range(config.n_layers)
        ])
        self.post_init()

    def forward(self, input_ids=None, **kwargs):
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        input_ids: torch.Tensor
            Input tensor of nucleotide tokens.
        """
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return BaseModelOutput(last_hidden_state=x)


class ConvNetOnlyMLMHead(nn.Module):
    """A head for masked language modeling."""
    def __init__(
        self,
        config,
    ):
        """
        Build a head for masked language modeling.
        This is Linear -> GELU -> LayerNorm -> Linear
        decoder that takes the hidden state of the model as input.

        Parameters
        ----------
        config: ConvNetConfig
            Configuration for the model.
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(self, hidden_state):
        """
        Perform a forward pass through the head.

        Parameters
        ----------
        hidden_state: torch.Tensor
            Hidden state of the model.
        
        Returns
        -------
        torch.Tensor
            Logits for each token in the vocabulary.
        """
        return self.decoder(hidden_state)


class ConvNetForMaskedLM(ConvNetPreTrainedModel):
    """A ResNet with dilated convolutions and a head for masked language modeling."""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        """
        Build a ResNet with dilated convolutions and a head for masked language modeling.

        Parameters
        ----------
        config: ConvNetConfig
            Configuration for the model.
        """
        super().__init__(config)
        self.config = config
        self.model = ConvNetModel(config)
        self.cls = ConvNetOnlyMLMHead(config)
        self.post_init()

    def forward(self, input_ids=None, labels=None, return_last_hidden_state = True, **kwargs):
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        input_ids: torch.Tensor
            Input tensor of nucleotide tokens with mask tokens.
        labels: torch.Tensor
            Input tensor of nucleotide tokens without mask tokens.
        return_last_hidden_state: bool
            Whether to return the last hidden state of the model.

        Returns
        -------
        transformers.modeling_outputs.MaskedLMOutput
            Output of the model.
        """
        hidden_state = self.model(input_ids=input_ids, **kwargs).last_hidden_state
        logits = self.cls(hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states = hidden_state if return_last_hidden_state else None
        )