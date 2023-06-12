import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput
from typing import List, Tuple
import math
import warnings

class AWDLSTMConfig(PretrainedConfig):
    model_type = "AWDLSTM"

    def __init__(
        self,
        vocab_size: int = 7,
        hidden_size: int = 512,
        # NOTE that hidden_size is not used when we run a single-layer LSTM.
        # as the decoder weights are tied with the embeddings, the LSTM needs to return vectors that have the same dimension as the embeddings.
        input_size: int = 64,
        num_hidden_layers: int = 1,
        initializer_range: float = 0.02,
        dropout_prob: float = 0.4,
        hidden_dropout_prob: float = 0.3,
        embedding_dropout_prob: float = 0.1,
        input_dropout_prob: float = 0.65,
        weight_dropout_prob: float = 0.5,
        beta: float = 1 ,
        alpha: float = 2,
        reset_token_id: int = None,
        bidirectional = False,
        batch_first: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.input_dropout_prob = input_dropout_prob
        self.weight_dropout_prob = weight_dropout_prob
        self.beta = beta
        self.alpha = alpha
        self.reset_token_id = reset_token_id
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.initializer_range = initializer_range


class AWDLSTMPreTrainedModel(PreTrainedModel):
    config_class = AWDLSTMConfig
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


class OneHotEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return F.one_hot(x, num_classes=self.hidden_size).float()


class LockedDropout(nn.Module):
    '''
    Dropout for the same inputs at each call
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class WeightDrop(nn.Module):
    """
    from https://github.com/a-martyn/awd-lstm/blob/master/model/net.py
    This only works for the custom lstm cell. If using default LSTM, 
    use this https://github.com/fastai/fastai2/blob/master/fastai2/text/models/awdlstm.py#L29
    Adapted to also handle h2h_reverse weights for bidirecional lstm.
    _____
    A module that wraps an LSTM cell in which some weights will be replaced by 0 during training.
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/text/models.py
    
    Initially I implemented this by getting the models state_dict attribute, modifying it to drop
    weights, and then loading the modified version with load_state_dict. I had to abandon this 
    approach after identifying it as the source of a slow memory leak.
    """
    def __init__(self, module:nn.Module, weight_p:float):
        super().__init__()
        self.module, self.weight_p = module, weight_p
        
        self.bidirectional = self.module.bidirectional

        #Makes a copy of the weights of the selected layers.
        w = getattr(self.module.h2h, 'weight')
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        self.module.h2h._parameters['weight'] = F.dropout(w, p=self.weight_p, training=False)

        if self.bidirectional:
            w_rev = getattr(self.module.h2h_reverse, 'weight')
            self.register_parameter('weight_raw_rev', nn.Parameter(w_rev.data))
            self.module.h2h_reverse._parameters['weight'] = F.dropout(w_rev, p=self.weight_p, training=False)


    def _setweights(self):
        "Apply dropout to the raw weights."
        raw_w = getattr(self, 'weight_raw')
        self.module.h2h._parameters['weight'] = F.dropout(raw_w, p=self.weight_p, training=self.training)

        if self.bidirectional:
            raw_w_rev = getattr(self, 'weight_raw_rev')
            self.module.h2h_reverse._parameters['weight'] = F.dropout(raw_w_rev, p=self.weight_p, training=self.training)


    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    # def reset(self):
    #     raw_w = getattr(self, 'weight_raw')
    #     self.module.h2h._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)

    #     if self.bidirectional:
    #         raw_w_rev = getattr(self, 'weight_raw_rev')
    #         self.module.h2h_reverse._parameters[layer] = F.dropout(raw_w_rev, p=self.weight_p, training=False)

    #     if hasattr(self.module, 'reset'): self.module.reset()


class LSTMCell(nn.Module):
    """
    LSTM cell to support resetting the hidden state when a end of sequence token is encountered.
    Cannot do that with pytorch default LSTM, as sequence steps are not accessible there.
    based on https://github.com/a-martyn/awd-lstm/blob/master/model/net.py
    Assume seq_len first dimension.
    """

    def __init__(self, input_size, output_size, bias=True, dropout =0, bidirectional = False, reset_token_id: int = -10000, batch_first: bool = False):
        super(LSTMCell, self).__init__()
        
        # Contains all weights for the 4 linear mappings of the input x
        # e.g. Wi, Wf, Wo, Wc
        self.i2h = nn.Linear(input_size, 4*output_size, bias=bias)
        # Contains all weights for the 4 linear mappings of the hidden state h
        # e.g. Ui, Uf, Uo, Uc
        self.h2h = nn.Linear(output_size, 4*output_size, bias=bias)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.i2h_reverse = nn.Linear(input_size, 4*output_size, bias=bias)
            self.h2h_reverse = nn.Linear(input_size, 4*output_size, bias=bias)
        self.output_size = output_size
        self.reset_token_id = reset_token_id
        self.reset_parameters()

        # code that sometime maybe. for now just protect against misuse.
        if batch_first:
            raise NotImplementedError('Only (seq_len, batch_size, dim) inputs supported.')

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            

    def _cell_step(self, x, hidden, tokens = None):
        '''
        performs one lstm update step.
        tokens: token_ids of the previous step. 
        If previous step had a reset_token, hidden state for this batch element is reset to 0 before the update.
        '''
        # unpack tuple (recurrent activations, recurrent cell state)
        #---- single cell time step operation
        h, c = hidden
        # reset the hidden states when and end of sequence token is encountered
        if tokens != None: #explicit check because of tensor behaviour
            idx = torch.where(tokens == self.reset_token_id)[0] #indices that are a reset_id token
            h[idx,:] = 0
            c[idx,:] = 0
        # Linear mappings : all four in one vectorised computation
        preact = self.i2h(x) + self.h2h(h)

        # Activations
        i = torch.sigmoid(preact[:, :self.output_size])                      # input gate
        f = torch.sigmoid(preact[:, self.output_size:2*self.output_size])    # forget gate
        g = torch.tanh(preact[:, 3*self.output_size:])                       # cell gate
        o = torch.sigmoid(preact[:, 2*self.output_size:3*self.output_size])  # output gate


        # Cell state computations: 
        # calculates new long term memory state based on proposed updates c_T
        # and input and forget gate states i_t, f_t
        c_t = torch.mul(f, c) + torch.mul(i, g)

        # Output
        h_t = torch.mul(o, torch.tanh(c_t))
        return h_t, c_t

    def _split_hidden_state(self, hidden_state: List[Tuple[torch.tensor, torch.tensor]]) -> Tuple[Tuple, Tuple]:
        '''Split concatenated hidden states for bidirectional model'''
        h_fwd, h_bwd = torch.chunk(hidden_state[0], 2, dim=-1)
        c_fwd, c_bwd = torch.chunk(hidden_state[1], 2, dim=-1)

        return (h_fwd, c_fwd), (h_bwd, c_fwd)

    def forward(self, input: torch.tensor, hidden_state: Tuple[torch.tensor, torch.tensor] = None, input_tokens = None):
        '''
        input: input tensor
        hidden_state: (h_t, c_t) tuple for inital hidden state
        input_tokens: Original input before embedding, used to reset the hidden state on eos tokens
        '''
        
        if self.bidirectional:
            #split the hidden state
            hidden_state, hidden_state_reverse = self._split_hidden_state(hidden_state)
                    
        output_list = []
        for t in range(input.size(0)):
            inp = input[t,:,:]
            
            h, c = hidden_state
            #squeeze and unsqueeze ops needed to be compatible with default lstm cell
            if input_tokens is not None:
                previous_tokens = (input_tokens[t-1,:] if t>1 else None)
                h_t, c_t = self._cell_step(inp, (h.squeeze(0), c.squeeze(0)), previous_tokens)
            else:
                h_t, c_t = self._cell_step(inp, (h.squeeze(0), c.squeeze(0)))
            hidden_state = (h_t.unsqueeze(0), c_t.unsqueeze(0)) #set new hidden state
            output_list.append(h_t)

        output = torch.stack(output_list)

        if self.bidirectional:
            input_reverse = torch.flip(input, [0])

            if input_tokens is not None:
                input_tokens_reverse = torch.flip(input_tokens, [0])
             #split hidden states

            output_list = []
            output_list_reverse = []
            for t in range(input.size(0)):
                inp = input[t,:,:]
                inp_reverse = input[t,:,:]

                h,c = hidden_state
                h_reverse, c_reverse = hidden_state_reverse

                if input_tokens is not None:
                    previous_tokens_reverse = (input_tokens_reverse[t-1,:] if t>1 else None)
                    h_t_reverse, c_t_reverse = self._cell_step(inp_reverse, (h_reverse.squeeze(), c_reverse.squeeze()), previous_tokens_reverse)
                else:
                    h_t_reverse, c_t_reverse = self._cell_step(inp_reverse, (h_reverse.squeeze(), c_reverse.squeeze()))
                hidden_state_reverse = (h_t_reverse.unsqueeze(0), c_t_reverse.unsqueeze(0))
                output_list_reverse.append(h_t_reverse)
            output_reverse = torch.stack(output_list_reverse)

            output = torch.cat([output,torch.flip(output_reverse, [0])], dim =-1) #reverse so positions match, then concat along feature dim
            h_cat = torch.cat([hidden_state[0], hidden_state_reverse[0]], dim =-1)
            c_cat = torch.cat([hidden_state[1], hidden_state_reverse[1]], dim =-1)
            hidden_state = (h_cat, c_cat)

        return output, hidden_state


class AWDLSTM(nn.Module):
    '''
    Multi-layer AWD-LSTM Model.
    Configuration (pass as `ProteinAWDLSTMConfig` object):
        num_layers:             number of AWD-LSTM layers
        input_size:             size of the inputs to the model (in most cases, size of the embedding layer). Also size of the output.
        hidden_size:            hidden size of the LSTM
        hidden_dropput_prob:    Dropout applied to the output between the LSTM layers
        input_dropout_prob:     Dropout applied to the input before passing through LSTM layers
        dropout_prob:           Dropout applied to final output after LSTM layers
        bidirectional:          Make the LSTM layers bidirectional. Output of model will be 2*input_size, forward and reverse hidden states concatenated.
    Args:
        config:                 ProteinAWDLSTMConfig object
        is_LM:                  Flag to also return outputs without hidden_dropout applied to them. Needed in LM for activation regularization.
        (type_2)                Hardcoded. Type 1 lstm concatenates after each layer (does not work with next token prediction). Type2 means two stacks of LSTMs
                                that are concatenated after the final layer.
    '''
    def __init__(self, config, is_LM: bool):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.input_dropout_prob = config.input_dropout_prob
        self.dropout_prob = config.dropout_prob #for consistency with original, output dropout would be more fitting
        self.bidirectional = config.bidirectional
        self.locked_dropout = LockedDropout() #Same instance reused everywhere
        self.is_LM = is_LM
        self.type_2 = False #this changes the bidirectional LSTM implementation. Type 2 only works with masked LM.

        #setup LSTM cells
        if self.bidirectional and self.type_2: # this is type2 biLSTM, where the outputs are concatenated between the layers. Does not work for LM.
            lstm = [LSTMCell(config.input_size if l == 0 else config.hidden_size *2, 
                            config.hidden_size if l != self.num_layers - 1 else config.input_size, 
                            1, 
                            dropout=0, 
                            bidirectional= config.bidirectional, 
                            reset_token_id= config.reset_token_id) 
                    for l in range(self.num_layers)]
            if config.weight_dropout_prob:
                lstm = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm]
                self.lstm = nn.ModuleList(lstm)
        elif self.bidirectional: #type 1 bidirectionality, two separate stacks of LSTMs
            lstm = [LSTMCell(config.input_size if l == 0 else config.hidden_size, 
                config.hidden_size if l != self.num_layers - 1 else config.input_size, 
                1, 
                dropout=0, 
                reset_token_id= config.reset_token_id) 
                for l in range(self.num_layers)]
            lstm_rev = [LSTMCell(config.input_size if l == 0 else config.hidden_size, 
                config.hidden_size if l != self.num_layers - 1 else config.input_size, 
                1, 
                dropout=0, 
                reset_token_id= config.reset_token_id) 
                for l in range(self.num_layers)]
            if config.weight_dropout_prob:
                lstm = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm]
                self.lstm = nn.ModuleList(lstm)
                lstm_rev = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm_rev]
                self.lstm_rev = nn.ModuleList(lstm_rev)
        else: 
            lstm = [LSTMCell(config.input_size if l == 0 else config.hidden_size, 
                    config.hidden_size if l != self.num_layers - 1 else config.input_size, 1, dropout=0, 
                    reset_token_id= config.reset_token_id) for l in range(self.num_layers)]
            if config.weight_dropout_prob:
                lstm = [WeightDrop(layer, config.weight_dropout_prob) for layer in lstm]
            self.lstm = nn.ModuleList(lstm)

    def forward(self, inputs, mask = None, hidden_state: List[Tuple[torch.Tensor, torch.Tensor]] = None, input_ids = None):
        '''
        inputs: (seq_len x batch_size x embedding_size)
        hidden_state: output from previous forward pass
        input_ids: original token ids to reset the hidden state
        returns:
            last layer output (all in format of default pytorch lstm)
            all layers hidden states (list)
            all layer outputs before droupout
            all layer outputs after dropout
        '''
        if  hidden_state is None:
            hidden_state = self._init_hidden(inputs.size(1), inputs)
        if self.bidirectional and not self.type_2:
            #list of (h,c) tuples with len = n_layers
            hidden_state_cat = hidden_state
            hidden_state, hidden_state_rev = [], []
            for layer_hidden in hidden_state_cat:
                hs_fwd, hs_bwd = torch.chunk(layer_hidden[0], 2, dim = -1)
                cs_fwd, cs_bwd = torch.chunk(layer_hidden[1], 2, dim = -1)
                hidden_state.append((hs_fwd, cs_fwd))
                hidden_state_rev.append((hs_bwd, cs_bwd))


        outputs_before_dropout = []
        hidden_states = []

        inputs = self.locked_dropout(inputs, self.input_dropout_prob)

        for i, layer in enumerate(self.lstm): #if self.is_LM else enumerate(self.lstm[:-1])):

            output, new_hidden_state = layer(inputs, hidden_state[i], input_ids)
            outputs_before_dropout.append(output)
            hidden_states.append(new_hidden_state)
            #apply dropout to hidden states
            if i != (self.num_layers if self.is_LM else self.num_layers-1 ):
                output = self.locked_dropout(output, self.hidden_dropout_prob)

            inputs = output
        
        output = self.locked_dropout(output, self.dropout_prob)

        if self.bidirectional and not self.type_2:
            outputs_before_dropout_rev = []
            hidden_states_rev = []
            inputs_rev = torch.flip(inputs, [0])
            input_ids_rev = None
            if input_ids is not None:
                input_ids_rev = torch.flip(input_ids, [0])
            
            for i, layer in enumerate(self.lstm_rev):
                output_rev, new_hidden_state_rev = layer(inputs_rev, hidden_state_rev[i], input_ids_rev)
                outputs_before_dropout_rev.append(output_rev)
                hidden_states_rev.append(new_hidden_state_rev)
                if i != (self.num_layers if self.is_LM else self.num_layers-1 ):
                    output = self.locked_dropout(output_rev, self.hidden_dropout_prob)
                inputs_rev = output_rev

            #concatenate all the forward and backward outputs and states.
            output= torch.cat([output, output_rev], dim = -1)
            outputs_before_dropout = [torch.cat([fwd,rev], dim =-1) for fwd, rev in zip(outputs_before_dropout, outputs_before_dropout_rev)]
            hidden_states = [(torch.cat([h, h_rev], dim =-1), torch.cat([c, c_rev], dim =-1)) for (h, c),(h_rev, c_rev) in zip(hidden_states, hidden_states_rev)]
            #hidden_states : List of tuples

        return output, hidden_states, outputs_before_dropout
    
    def _init_hidden(self, batch_size, dtype_tensor):
        '''
        Create initial all zero hidden states for the lstm layers.
        dtype_tensor is some tensor that tells us which device and dtype we need.
        '''
        weight = dtype_tensor
        states = [(weight.new_zeros(1, batch_size, self.hidden_size if l != self.num_layers - 1 else self.input_size),
                    weight.new_zeros(1, batch_size, self.hidden_size if l != self.num_layers - 1 else self.input_size)) for l in range(self.num_layers)]
        if self.bidirectional and not self.type_2: # *2 because of concatenation of forward and reverse states
            states = [(weight.new_zeros(1, batch_size, self.hidden_size*2 if l != self.num_layers - 1 else self.input_size*2),
                    weight.new_zeros(1, batch_size, self.hidden_size*2 if l != self.num_layers - 1 else self.input_size*2)) for l in range(self.num_layers)]
        return states


class AWDLSTMModel(AWDLSTMPreTrainedModel):
    def __init__(
        self,
        config,
        is_LM = False,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.input_size)#OneHotEmbedding(config.input_size)
        self.embedding_dropout_prob = config.embedding_dropout_prob
        self.batch_first = config.batch_first
        self.reset_token_id = config.reset_token_id

        self.encoder = AWDLSTM(config, is_LM=is_LM)

        self.is_LM = is_LM
        
        self.init_weights()
        self.post_init()

    def forward(self, input_ids=None, input_mask=None, hidden_state=None, **kwargs):
        if self.batch_first:
            input_ids = input_ids.transpose(0,1)
            if input_mask is not None:
                input_mask = input_mask.transpose(0,1)
            if hidden_state is not None:
                hidden_state = [(x[0].transpose(0,1), x[1].transpose(0,1)) for x in hidden_state]


        if input_mask is None:
            input_mask = torch.ones_like(input_ids)


        x = self.embedding(input_ids)

        if self.reset_token_id is not None:
            encoder_outputs = self.encoder(x, input_mask, hidden_state, input_ids)
        else:
            encoder_outputs = self.encoder(x, input_mask, hidden_state)

        output, hidden_state, outputs_raw = encoder_outputs
        # output = last layer output tensor
        # hidden_state = last (h,c) states of each layer to keep track of the cell state
        # outputs_raw = list of output tensors from each layer, outputs_raw[-1] is eq. to output.

        # transpose back if batch_first
        if self.batch_first:
            output = output.transpose(0,1)
            hidden_state = [(x[0].transpose(0,1), x[1].transpose(0,1)) for x in hidden_state]
            outputs_raw = [x.transpose(0,1) for x in outputs_raw]

            # outputs_before_dropout = [torch.cat([fwd,rev], dim =-1) for fwd, rev in zip(outputs_before_dropout, outputs_before_dropout_rev)]
            # hidden_states = [(torch.cat([h, h_rev], dim =-1), torch.cat([c, c_rev], dim =-1)) for (h, c),(h_rev, c_rev) in zip(hidden_states, hidden_states_rev)]

        # hacky way to allow me to use the old training class without breaking API downstream.
        if self.is_LM:
            return output, hidden_state, outputs_raw
        
        
        return BaseModelOutput(last_hidden_state=output, hidden_states=outputs_raw)

class AWDLSTMModelForInference(AWDLSTMPreTrainedModel):
    '''Use this model to run inference with a pretrained model.'''
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AWDLSTMModel(config = config, is_LM = False)

        self.batch_first = config.batch_first

        self.init_weights()


    def forward(self, input_ids, input_mask=None, hidden_state=None):

        outputs = self.encoder(input_ids, input_mask, hidden_state)
        return outputs

        



class AWDLSTMForLM(AWDLSTMPreTrainedModel):
    '''
    Model to run the original AWD-LSTM pretraining strategy.
    - reuse the hidden state
    - activation regularization
    Supports bidirectional training. As we want the loss to be interpretable as perplexity, it
    performs next token prediction seperately for the forward and backward LSTM outputs. This 
    means that also targets_reverse need to be given, as the tokens the backwards model predicts
    are data[train_idx -1: train_idx] as opposed to data[train_idx]
    WAIT A SEC THIS IS ONLY TRUE FOR ANNOYING TBBT TRAINING, IN GENERAL [START-1] DOESNT EVEN EXIST
    '''
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AWDLSTMModel(config = config, is_LM = True)
        self.decoder = nn.Linear(config.input_size, config.vocab_size)
        #some people say this does not work, but is in original code
        self.decoder.weight = self.encoder.embedding.weight
        self.alpha = config.alpha
        self.beta = config.beta

        self.batch_first = config.batch_first

        self.init_weights()

        # we need buffers for each layer + bidirectional.
        for l in range(self.encoder.config.num_hidden_layers):
            self.register_buffer(f'last_hidden_state_{l}', None)
            self.register_buffer(f'last_cell_state_{l}', None)

    
    def _set_last_hidden_state(self, values):

        for idx, v in enumerate(values):
            setattr(self, f'last_hidden_state_{idx}', v[0].detach())
            setattr(self, f'last_cell_state_{idx}', v[1].detach())
    
    def _get_last_hidden_state(self):
        out = []
        for idx in range(self.encoder.config.num_hidden_layers):
            h = getattr(self, f'last_hidden_state_{idx}')
            if h is None:
                return None #shortcut in first step or when not using mem.
            c = getattr(self, f'last_cell_state_{idx}')
            out.append([h,c])
        
        return out

    def _regularize_loss(self, loss: torch.Tensor, sequence_output: torch.Tensor, last_output: torch.Tensor) -> torch.Tensor:
        if self.alpha:
            ar = self.alpha * sequence_output.pow(2).mean()
            loss += ar
        if self.beta:
             #regularization on the difference between steps, computed for the last layer output only
            #squared difference h_after_step - h_before_step
            if self.batch_first:
                tar = self.beta * (last_output[:,1:] - last_output[:,:-1]).pow(2).mean()
            else:
                tar = self.beta * (last_output[1:] - last_output[:-1]).pow(2).mean()
            loss += tar
        return loss

    def forward(self, input_ids, input_mask=None, labels=None, output_hidden_states=False):
        targets=labels
        outputs = self.encoder(input_ids, input_mask, self._get_last_hidden_state())
        sequence_output, hidden_state, raw_outputs = outputs[:3] #raw_outputs: list of [seq_len x batch_size x output_dim] tensors
        last_layer_raw_output = raw_outputs[-1]

        sequence_output_full = sequence_output #for regularization
        last_layer_raw_output_full = last_layer_raw_output #for regularization
        
        if self.encoder.encoder.bidirectional:
            #if the LM is bidirectional, decoding is done twice.
            sequence_output, sequence_output_reverse = torch.chunk(sequence_output,2, dim =-1)
            last_layer_raw_output, last_layer_raw_output_reverse = torch.chunk(last_layer_raw_output, 2, dim = -1)
            prediction_scores = self.decoder(sequence_output)
            prediction_scores_reverse = self.decoder(sequence_output_reverse)
            outputs = (torch.cat([prediction_scores, prediction_scores_reverse], dim =-1), )
        else:
            prediction_scores = self.decoder(sequence_output)
            outputs = (prediction_scores,)
        
        # keep hidden state to reuse in next batch.
        self._set_last_hidden_state(hidden_state)
        

        # these are the logits.
        prediction_scores = prediction_scores.contiguous()

        if targets is not None:
            targets_fwd = targets[:, 1:] if self.batch_first else targets[1:]
            scores_fwd = prediction_scores[:, :-1] if self.batch_first else prediction_scores[:-1]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # changed ignore_index from -1 -> -100
            raw_loss = loss_fct(
                scores_fwd.contiguous().view(-1, self.config.vocab_size), targets_fwd.contiguous().view(-1))
            if self.encoder.encoder.bidirectional:
                targets_rev = targets[:, :-1] if self.batch_first else targets[:-1]
                scores_rev = prediction_scores_reverse[:,1:] if self.batch_first else prediction_scores_reverse[1:]
                raw_loss_reverse = loss_fct(
                    scores_rev.contiguous().view(-1, self.config.vocab_size), targets_rev.contiguous().view(-1))

                raw_loss = (raw_loss + raw_loss_reverse) /2.

            lm_loss = self._regularize_loss(raw_loss, sequence_output_full, last_layer_raw_output_full)
            outputs = (lm_loss, raw_loss) + outputs
        else:
            lm_loss = None
        
        return CausalLMOutput(loss=lm_loss, logits=prediction_scores, hidden_states=sequence_output if output_hidden_states else None)
