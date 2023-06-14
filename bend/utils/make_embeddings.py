'''
Utilities to compute embeddings from various models.  
The following models are supported:

- GPN
- DNABERT
- Species-aware DNA LM
- Nucleotide Transformer

Usage: either as functions or as classes.
```
embed_gpn(sequences)
embed_dnabert(sequences, 'checkpoints/dnabert/3-new-12w-0/', kmer=3)
embed_species_aware(sequences, 'venturia_inaequalis', 'checkpoints/species_aware_ssm.pt')
embed_revolution(sequences, 'checkpoints/model_ve_1hot_steps_49_mask_rate_0.3_mask_ratio_0.8_len_10000_dim1_64_dim2_32.pt')
embed_nucleotide_transformer(sequences)

embedder = GPNEmbedder()
embedder.embed(sequences)
```
'''
import torch
import numpy as np
from typing import List
from functools import partial
from tqdm.auto import tqdm
from transformers import logging
logging.set_verbosity_error()


# TODO graceful auto downloading solution for everything that is hosted in a nice way
# https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


##
## GPN https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1
##

class BaseEmbedder():
    def __init__(self, *args, **kwargs):
        self.load_model(*args, **kwargs)

    def load_model(*args, **kwargs):
        raise NotImplementedError
    
    def embed(*args, **kwargs):
        raise NotImplementedError

 

class GPNEmbedder(BaseEmbedder):

    def load_model(self):
        try:
            import gpn.model
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('GPN requires gpn. Install with: pip install git+https://github.com/songlab-cal/gpn.git')

        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('GPN requires transformers. Install with: pip install transformers')


        # TODO change type to ConvNetModel to make this work.
        self.model = AutoModel.from_pretrained("gonzalobenegas/gpn-arabidopsis")
        self.tokenizer = AutoTokenizer.from_pretrained("gonzalobenegas/gpn-arabidopsis")

        self.model.to(device)
        self.model.eval()

    def embed(self, sequences: List[str]) -> List[np.ndarray]:
        '''Run the GPN model https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1'''

        embeddings = []
        with torch.no_grad():
            for seq in tqdm(sequences):
                input_ids = self.tokenizer(seq, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                input_ids = input_ids.to(device)
                embedding = self.model(input_ids=input_ids).last_hidden_state
                
                embeddings.append(embedding.detach().cpu().numpy())

        return embeddings



##
## Species-aware DNA language modeling https://www.biorxiv.org/content/10.1101/2023.01.26.525670v1
##

class SpeciesAwareEmbedder(BaseEmbedder):

    def load_model(self, checkpoint_path: str = 'data/models/species_aware_ssm.ckpt') -> List[np.ndarray]:
        '''https://www.biorxiv.org/content/10.1101/2023.01.26.525670v1'''

        # copied their code - repo not installable as package.
        # also had to sanitize checkpoint, expected src to be available for unpickling
        # x = torch.load('../DNA-LM/data/models/species_aware_ssm.ckpt', map_location='cpu')
        # # x['hyper_parameters']['net'] # has the problem - this is a pickled DSSResNet
        # state_dict = x['state_dict'] #net.encoder.weight', 'net.encoder.bias' ...
        # torch.save(state_dict, 'species_aware_ssm.pt') # 4.2mb size

        # https://github.com/DennisGankin/species-aware-DNA-LM/blob/main/configs/model/species_dss_weights.yaml
        from models.species_aware.spec_dss import DSSResNetEmb, SpecAdd

        from models.species_aware.motif_model_species_weights import MotifModule
        

        # https://github.com/DennisGankin/species-aware-DNA-LM/blob/main/src/datamodules/__init__.py#L228
        from models.species_aware.utils import spec_dict

        # try importing requirements here to raise errors?
        try:
            import opt_einsum
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Species-aware requires opt_einsum. Install with: pip install opt_einsum')
        try:
            import omegaconf
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Species-aware requires omegaconf. Install with: pip install omegaconf')



        species_encoder = SpecAdd(embed=True, encoder='label', d_model=128)
        
        resnet = DSSResNetEmb(
            species_encoder = species_encoder,
            d_input = 5,
            d_output = 5,
            d_model = 128,
            n_layers = 4,
            dropout = 0.1,
            embed_before = True    
        )

        self.model = MotifModule(net = resnet)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(device)
        self.model.eval()

        self.spec_dict = spec_dict

    def embed(self, sequences: List[str], species) -> List[np.ndarray]:
        embeddings= []
        with torch.no_grad():
            for seq in tqdm(sequences):

                sequence_onehot, sequence = self._species_aware_encode(seq)

                spec = self.spec_dict[species]
                spec = torch.LongTensor(np.array([spec]))

                # dummy batch dims
                # Input x is shape (B, d_input, L)
                sequence_onehot = sequence_onehot.unsqueeze(0).to(device)
                spec = spec.unsqueeze(0).to(device)

                x, emb = self.model(sequence_onehot, spec)
                embeddings.append(emb['seq_embedding'].detach().cpu().numpy())


        return embeddings


    @staticmethod
    def _one_hot_encode(seq):
        """
        Computes class labels and one hot encoding of DNA sequence passed as string or binary string
        """
        # allow string and binary string
        mapping = dict(zip(b"ACGTN", range(5)))
        mapping.update(dict(zip("ACGTN", range(5))))
        seq2 = []
        for i in seq:
            try:
                seq2.append(mapping[i])
            except:
                seq2.append(mapping["N"])
        # return as sequence of class labels and as one hot encoded sequence
        seq2 = np.array(seq2)
        return seq2, np.eye(5)[seq2]


    def _species_aware_encode(self, seq):
        '''Adapted from encode_and_pad_test() - don't need masking and motifs.'''

        seq_labels, seq_one_hot = self._one_hot_encode(seq)

        # do not mask for testing 
        # create same legth zeros
        motif_target_seq = np.zeros(len(seq))

        # make sure x has type float and labels type long 
        x = torch.from_numpy(seq_one_hot.transpose()).float()
        y = torch.from_numpy(seq_labels.transpose()).long()

        return x, y 


##
## DNABert https://doi.org/10.1093/bioinformatics/btab083
##

class DNABertEmbedder(BaseEmbedder):

    def load_model(self, 
                   dnabert_path: str = '../../external-models/DNABERT/', 
                   kmer: int = 3, ):
        try:
            from transformers import (
                BertModel,
                BertConfig,
                BertTokenizer,
            )
        except ModuleNotFoundError as e:
            print(e)
            print('DNABert requires transfomers. Install with: pip install transformers')

        dnabert_path = f'{dnabert_path}/DNABERT{kmer}/'

        config = BertConfig.from_pretrained(dnabert_path)
        self.tokenizer = BertTokenizer.from_pretrained(dnabert_path)
        self.bert_model = BertModel.from_pretrained(dnabert_path, config=config)
        self.bert_model.to(device)
        self.bert_model.eval()

        self.kmer = kmer

    def embed(self, sequences: List[str], disable_tqdm: bool = False):
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences, disable=disable_tqdm):
                sequence = [sequence]
                kmers = self._seq2kmer_batch(sequence, self.kmer)
                model_input = self.tokenizer.batch_encode_plus(kmers, add_special_tokens=False, 
                                                               max_length=len(sequence[0]), return_tensors='pt', 
                                                               padding='max_length')["input_ids"]
                if model_input.shape[1] > 512:
                    model_input = torch.split(model_input, 512, dim=1)
                    output = []
                    for chunk in model_input: 
                        output.append(self.bert_model(chunk.to(device))[0].detach().cpu())
                    output = torch.cat(output, dim=1)
                else:
                    output = self.bert_model(model_input.to(device))
                embedding = output[0].detach().cpu().numpy()
                embeddings.append(embedding)

        return embeddings

    @staticmethod
    def _seq2kmer(seq, k):
        """
        Convert original sequence to kmers
        
        Arguments:
        seq -- str, original sequence.
        k -- int, kmer of length k specified.
        
        Returns:
        kmers -- str, kmers separated by space
        """
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        return kmers

    def _seq2kmer_batch(self, batch, k=3, step_size=1, kmerise=True):
        return list(map(partial(self._seq2kmer, k = k), batch))

##
## Revolution https://www.biorxiv.org/content/10.1101/2023.01.30.526193v1
##

REVOLUTION_BASE_TO_INDEX = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3,
    'a': 0, 'c': 1, 'g': 2, 't': 3,
}

class RevolutionEmbedder(BaseEmbedder):

    def load_model(self, path):
        from models.revolution.model_revolution import Revolution_Conv

        # inferred the config from the checkpoint shapes.
        # model_ve_1hot_steps_49_mask_rate_0.3_mask_ratio_0.8_len_10000_dim1_64_dim2_32.pt
        n_layers = 13
        self.model = Revolution_Conv(5, [64]*n_layers, 128, ks=3)
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.to(device)
        self.model.eval()

    def embed(self, sequences: List[str]):
        embeddings = []
        with torch.no_grad():
            for sequence in tqdm(sequences):

                inp = self._revolution_one_hot_encode(sequence)
                inp = inp.transpose(1,0).unsqueeze(0)  # first layer is Conv1d: (N,C in,L) 

                emb = self.model(inp.to(device))
                embeddings.append(emb.detach().cpu().numpy())

        return embeddings

    @staticmethod
    def _revolution_one_hot_encode(sequence, dim=5):

        encoding = np.zeros((len(sequence), 5), dtype=np.float32)
        n_fill = np.divide(1, dim, dtype=np.float32)

        for index in range(len(sequence)):
            base = sequence[index]
            if base in REVOLUTION_BASE_TO_INDEX:
                encoding[index, REVOLUTION_BASE_TO_INDEX[base]] = 1
            else:
                encoding[index, :] = n_fill

        return torch.from_numpy(encoding)

class NucleotideTransformerEmbedder(BaseEmbedder):

    def load_model(self, model_name):
        try:
            from nucleotide_transformer.tokenizers import FixedSizeNucleotidesKmersTokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Nucleotide transformer requires nucleotide-transformer. Install with: pip install git+https://github.com/instadeepai/nucleotide-transformer.git')
        try:
            from transformers import AutoTokenizer, AutoModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Nucleotide transformer requires transformers. Install with: pip install transformers')

        # Get pretrained model
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def embed(self, sequences: List[str], disable_tqdm: bool = False, return_cls_token: bool = False):
        '''Tokenizes and embeds sequences. CLS token is removed from the output.'''
        
        cls_tokens = []
        embeddings = []
        
        with torch.no_grad():
            for n, s in enumerate(tqdm(sequences, disable=disable_tqdm)):
                #print('sequence', n)
                s_chunks = [s[chunk : chunk + 5994] for chunk in  range(0, len(s), 5994)] # split into chunks 
                embedded_seq = []
                cls_seq = []
                for n_chunk, chunk in enumerate(s_chunks): # embed each chunk
                    tokens_ids = self.tokenizer(chunk, return_tensors = 'pt')['input_ids'].int().to(device)
                    if len(tokens_ids[0]) > 1000: # too long to fit into the model
                        split = torch.split(tokens_ids, 1000, dim=-1)
                        outs = [self.model(item)['last_hidden_state'].detach().cpu().numpy() for item in split]
                        outs = np.concatenate(outs, axis=1)
                    else:
                        outs = self.model(tokens_ids)['last_hidden_state'].detach().cpu().numpy() # get last hidden state
                    embedded_seq.append(outs[:,1:])
                    #print('chunk', n_chunk, 'chunk length', len(chunk), 'tokens length', len(tokens_ids[0]), 'chunk embedded shape', outs.shape)
                    cls_seq.append(outs[:,0])
                embeddings.append(np.concatenate(embedded_seq, axis=1)) 
                cls_tokens.append(np.concatenate(cls_seq, axis=0))
        if return_cls_token:
            return embeddings, cls_tokens

        return embeddings


class AWDLSTMEmbedder(BaseEmbedder):

    def load_model(self, model_path, tokenizer_path: str = '../tokenizers/tokenizer_extended/', **kwargs):

        from models.AWDLSTM import AWDLSTMModelForInference
        from transformers import AutoTokenizer


        # Get pretrained model
        self.model = AWDLSTMModelForInference.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def embed(self, sequences: List[str], disable_tqdm: bool = False):
        '''Tokenizes and embeds sequences. CLS token is removed from the output.'''
        embeddings = []
        with torch.no_grad():
            for s in tqdm(sequences, disable=disable_tqdm):

                input_ids = self.tokenizer(s, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                input_ids = input_ids.to(device)
                embedding = self.model(input_ids=input_ids).last_hidden_state
                
                embeddings.append(embedding.detach().cpu().numpy())
                # embeddings.append(embedding.detach().cpu().numpy()[:,1:])
            
        return embeddings
    
class ConvNetEmbedder(BaseEmbedder):
    def load_model(self, model_path, tokenizer_path = '../tokenizers/tokenizer_bare/', **kwargs):
        from transformers import AutoTokenizer, PretrainedConfig
        from transformers import logging
        logging.set_verbosity_error()
        try: 
            from models.DilatedCNN import ConvNetModel
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Cannot find Module ConvNetModel.')
        # load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except: 
            raise ValueError('tokenizer_path is not a valid path')
        
        
        self.model = ConvNetModel.from_pretrained(model_path).to(device).eval()
    
    def embed(self, sequences: List[str], disable_tqdm: bool = False):
        embeddings = [] 
        with torch.no_grad():
            for s in tqdm(sequences, disable=disable_tqdm):
                input_ids = self.tokenizer(s, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                input_ids = input_ids.to(device)
                embedding = self.model(input_ids=input_ids).last_hidden_state
                embeddings.append(embedding.detach().cpu().numpy())

        return embeddings
    
# backward compatibility

def embed_dnabert(sequences, path: str = '/z/home/frma/projects/DNA-LM/external-models/DNABERT/DNABERT3/', 
                  kmer: int = 3, disable_tqdm = False):
    return DNABertEmbedder(path, kmer).embed(sequences, disable_tqdm = disable_tqdm)

def embed_gpn(sequences):
    return GPNEmbedder().embed(sequences)

def embed_species_aware(sequences, species, path):
    return SpeciesAwareEmbedder(path).embed(sequences, species)

def embed_revolution(sequences, path):
    return RevolutionEmbedder(path).embed(sequences)

def embed_nucleotide_transformer(sequences, model_name):
    return NucleotideTransformerEmbedder(model_name).embed(sequences)

def embed_awdlstm(sequences, model_path, tokenizer_path, disable_tqdm = False, **kwargs):
    return AWDLSTMEmbedder(model_path, tokenizer_path, **kwargs).embed(sequences, disable_tqdm = disable_tqdm )

def embed_convnet(sequences, model_path, tokenizer_path, disable_tqdm = False, **kwargs):
    return ConvNetEmbedder(model_path, tokenizer_path, **kwargs).embed(sequences, disable_tqdm = disable_tqdm)


def embed_sequence(sequences : List[str], embedding_type : str = 'categorical', **kwargs):
    '''
    sequences : list of sequences to embed
    '''
    if not embedding_type:
        return sequences
    
    if embedding_type == 'categorical' or embedding_type == 'onehot':
        from .sequences import EncodeSequence
        encode_seq = EncodeSequence() 
        # embed to categorcal  
        sequence = []
        for seq in sequences:
            sequence.append(torch.tensor(encode_seq.transform_integer(seq)))
            return sequence
    # embed with nt transformer:   
    elif embedding_type == 'nt_transformer':
        # model name "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
        sequences, cls_token = embed_nucleotide_transformer(sequences, **kwargs)
        return sequences, cls_token
    # embed with GPN 
    # embed with DNAbert
    elif embedding_type == 'dnabert':
        sequences = embed_dnabert(sequences, disable_tqdm = True, **kwargs)
        # /z/home/frma/projects/DNA-LM/external-models/DNABERT/DNABERT3/
        # kmer = 3 
        return sequences
    # embed with own models. 
    elif embedding_type == 'awdlstm':
        sequences = embed_awdlstm(sequences, disable_tqdm = True, **kwargs)
        return sequences
    elif embedding_type == 'convnet':
        sequences = embed_convnet(sequences, disable_tqdm = True, **kwargs)
        return sequences

    return sequences

def test():
    '''Test that models run. Does not check that the output is correct.'''
    sequences =  ["ATGCCCTGGC", "AATACGGT"]

    embed_gpn(sequences)
    embed_dnabert(sequences, '../checkpoints/dnabert/3-new-12w-0/', kmer=3)
    embed_species_aware(sequences, 'venturia_inaequalis', '../checkpoints/species_aware_ssm.pt')
    embed_revolution(['A'*10000], '../checkpoints/model_ve_1hot_steps_49_mask_rate_0.3_mask_ratio_0.8_len_10000_dim1_64_dim2_32.pt')
    embed_nucleotide_transformer(sequences, "InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
    embed_awdlstm(sequences, '1m_steps_lr1e3_all_data/checkpoint-1000000', '../scripts/tokenizers/tokenizer_extended/')
    embed_convnet(sequences,  '/z/home/frma/projects/DNA-LM/trained-models/chromosome_split/human/checkpoint-436000/', '/z/home/frma/projects/DNA-LM/scripts/tokenizers/tokenizer_bare/')

#     odict_keys(['conv_net.0.conv1.weight', 'conv_net.0.conv2.weight', 'conv_net.0.batch_norm1.weight', 
#                 'conv_net.0.batch_norm1.bias', 'conv_net.0.batch_norm1.running_mean', 'conv_net.0.batch_norm1.running_var', 
#                 'conv_net.0.batch_norm1.num_batches_tracked', 'conv_net.0.batch_norm2.weight', 'conv_net.0.batch_norm2.bias', 
#                 'conv_net.0.batch_norm2.running_mean', 'conv_net.0.batch_norm2.running_var', 'conv_net.0.batch_norm2.num_batches_tracked', 
#                 'conv_net.0.res.weight', 'conv_net.0.res.bias', 'conv_net.1.conv1.weight', 'conv_net.1.conv2.weight', 'conv_net.1.batch_norm1.weight', 
#                 'conv_net.1.batch_norm1.bias', 'conv_net.1.batch_norm1.running_mean', 'conv_net.1.batch_norm1.running_var', 'conv_net.1.batch_norm1.num_batches_tracked', 
#                 'conv_net.1.batch_norm2.weight', 'conv_net.1.batch_norm2.bias', 'conv_net.1.batch_norm2.running_mean', 'conv_net.1.batch_norm2.running_var', 
#                 'conv_net.1.batch_norm2.num_batches_tracked', 'conv_net.2.conv1.weight', 'conv_net.2.conv2.weight', 'conv_net.2.batch_norm1.weight', 
#                 'conv_net.2.batch_norm1.bias', 'conv_net.2.batch_norm1.running_mean', 'conv_net.2.batch_norm1.running_var', 'conv_net.2.batch_norm1.num_batches_tracked', 
#                 'conv_net.2.batch_norm2.weight', 'conv_net.2.batch_norm2.bias', 'conv_net.2.batch_norm2.running_mean', 'conv_net.2.batch_norm2.running_var', 
#                 'conv_net.2.batch_norm2.num_batches_tracked', 'conv_net.3.conv1.weight', 'conv_net.3.conv2.weight', 'conv_net.3.batch_norm1.weight', 
#                 'conv_net.3.batch_norm1.bias', 'conv_net.3.batch_norm1.running_mean', 'conv_net.3.batch_norm1.running_var', 'conv_net.3.batch_norm1.num_batches_tracked', 
#                 'conv_net.3.batch_norm2.weight', 'conv_net.3.batch_norm2.bias', 'conv_net.3.batch_norm2.running_mean', 'conv_net.3.batch_norm2.running_var', 
#                 'conv_net.3.batch_norm2.num_batches_tracked', 'conv_net.4.conv1.weight', 'conv_net.4.conv2.weight', 'conv_net.4.batch_norm1.weight', 
#                 'conv_net.4.batch_norm1.bias', 'conv_net.4.batch_norm1.running_mean', 'conv_net.4.batch_norm1.running_var', 'conv_net.4.batch_norm1.num_batches_tracked', 'conv_net.4.batch_norm2.weight', 'conv_net.4.batch_norm2.bias', 
#                 'conv_net.4.batch_norm2.running_mean', 'conv_net.4.batch_norm2.running_var', 'conv_net.4.batch_norm2.num_batches_tracked', 'conv_net.5.conv1.weight', 'conv_net.5.conv2.weight', 'conv_net.5.batch_norm1.weight', 'conv_net.5.batch_norm1.bias', 
#                 'conv_net.5.batch_norm1.running_mean', 'conv_net.5.batch_norm1.running_var', 'conv_net.5.batch_norm1.num_batches_tracked', 'conv_net.5.batch_norm2.weight', 'conv_net.5.batch_norm2.bias', 'conv_net.5.batch_norm2.running_mean', 
#                 'conv_net.5.batch_norm2.running_var', 'conv_net.5.batch_norm2.num_batches_tracked', 'conv_net.6.conv1.weight', 'conv_net.6.conv2.weight', 'conv_net.6.batch_norm1.weight', 'conv_net.6.batch_norm1.bias', 'conv_net.6.batch_norm1.running_mean', 
#                 'conv_net.6.batch_norm1.running_var', 'conv_net.6.batch_norm1.num_batches_tracked', 'conv_net.6.batch_norm2.weight', 'conv_net.6.batch_norm2.bias', 'conv_net.6.batch_norm2.running_mean', 'conv_net.6.batch_norm2.running_var', 
#                 'conv_net.6.batch_norm2.num_batches_tracked', 'conv_net.7.conv1.weight', 'conv_net.7.conv2.weight', 'conv_net.7.batch_norm1.weight', 'conv_net.7.batch_norm1.bias', 'conv_net.7.batch_norm1.running_mean', 'conv_net.7.batch_norm1.running_var', 'conv_net.7.batch_norm1.num_batches_tracked', 'conv_net.7.batch_norm2.weight', 'conv_net.7.batch_norm2.bias', 'conv_net.7.batch_norm2.running_mean', 'conv_net.7.batch_norm2.running_var', 'conv_net.7.batch_norm2.num_batches_tracked', 'conv_net.8.conv1.weight', 'conv_net.8.conv2.weight', 
#                 'conv_net.8.batch_norm1.weight', 'conv_net.8.batch_norm1.bias', 'conv_net.8.batch_norm1.running_mean', 'conv_net.8.batch_norm1.running_var', 
#                 'conv_net.8.batch_norm1.num_batches_tracked', 'conv_net.8.batch_norm2.weight', 'conv_net.8.batch_norm2.bias', 'conv_net.8.batch_norm2.running_mean', 'conv_net.8.batch_norm2.running_var', 'conv_net.8.batch_norm2.num_batches_tracked', 
#                 'conv_net.9.conv1.weight', 'conv_net.9.conv2.weight', 'conv_net.9.batch_norm1.weight', 'conv_net.9.batch_norm1.bias', 'conv_net.9.batch_norm1.running_mean', 'conv_net.9.batch_norm1.running_var', 'conv_net.9.batch_norm1.num_batches_tracked', 
#                 'conv_net.9.batch_norm2.weight', 'conv_net.9.batch_norm2.bias', 'conv_net.9.batch_norm2.running_mean', 'conv_net.9.batch_norm2.running_var', 'conv_net.9.batch_norm2.num_batches_tracked', 'conv_net.10.conv1.weight', 'conv_net.10.conv2.weight', 
#                 'conv_net.10.batch_norm1.weight', 'conv_net.10.batch_norm1.bias', 'conv_net.10.batch_norm1.running_mean', 'conv_net.10.batch_norm1.running_var', 'conv_net.10.batch_norm1.num_batches_tracked', 'conv_net.10.batch_norm2.weight', 'conv_net.10.batch_norm2.bias', 
#                 'conv_net.10.batch_norm2.running_mean', 'conv_net.10.batch_norm2.running_var', 'conv_net.10.batch_norm2.num_batches_tracked', 'conv_net.11.conv1.weight', 'conv_net.11.conv2.weight', 'conv_net.11.batch_norm1.weight', 'conv_net.11.batch_norm1.bias', 
#                 'conv_net.11.batch_norm1.running_mean', 'conv_net.11.batch_norm1.running_var', 'conv_net.11.batch_norm1.num_batches_tracked', 'conv_net.11.batch_norm2.weight', 'conv_net.11.batch_norm2.bias', 'conv_net.11.batch_norm2.running_mean', 
#                 'conv_net.11.batch_norm2.running_var', 'conv_net.11.batch_norm2.num_batches_tracked', 'conv_net.12.conv1.weight', 'conv_net.12.conv2.weight', 'conv_net.12.batch_norm1.weight', 
#                 'conv_net.12.batch_norm1.bias', 'conv_net.12.batch_norm1.running_mean', 'conv_net.12.batch_norm1.running_var', 'conv_net.12.batch_norm1.num_batches_tracked', 'conv_net.12.batch_norm2.weight', 
#                 'conv_net.12.batch_norm2.bias', 'conv_net.12.batch_norm2.running_mean', 'conv_net.12.batch_norm2.running_var', 'conv_net.12.batch_norm2.num_batches_tracked'])