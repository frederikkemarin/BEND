'''
Utilities to compute embeddings from various models.  
The following models are supported:

- GPN
- DNABERT
- Nucleotide Transformer
- AWD-LSTM
- ResNet-LM

Usage: either as functions or as classes.
```


# get any of the embedders
embedder = GPNEmbedder()
embedder = DNABertEmbedder('path/to/checkpoint', kmer=6)
embedder = NucleotideTransformerEmbedder('checkpoint_name')
embedder = AWDLSTMEmbedder('path/to/checkpoint')
embedder = ConvNetEmbedder('path/to/checkpoint')

# embed
sequences =  ["ATGCCCTGGC", "AATACGGT"]
embedder.embed(sequences, disable_tqdm=True)
```
    
'''


import torch
import numpy as np
from typing import List
from functools import partial

from ..models import AWDLSTMModelForInference, ConvNetModel

from tqdm.auto import tqdm
from transformers import logging, BertModel, BertConfig, BertTokenizer, AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
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

    def load_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def embed(self, *args, **kwargs):
        raise NotImplementedError

 

class GPNEmbedder(BaseEmbedder):

    def load_model(self):
        try:
            import gpn.model
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('GPN requires gpn. Install with: pip install git+https://github.com/songlab-cal/gpn.git')


        self.model = AutoModel.from_pretrained("gonzalobenegas/gpn-arabidopsis")
        self.tokenizer = AutoTokenizer.from_pretrained("gonzalobenegas/gpn-arabidopsis")

        self.model.to(device)
        self.model.eval()

    def embed(self, sequences: List[str], disable_tqdm: bool = False) -> List[np.ndarray]:
        '''Run the GPN model https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1'''

        embeddings = []
        with torch.no_grad():
            for seq in tqdm(sequences, disable=disable_tqdm):
                input_ids = self.tokenizer(seq, return_tensors="pt", return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                input_ids = input_ids.to(device)
                embedding = self.model(input_ids=input_ids).last_hidden_state
                
                embeddings.append(embedding.detach().cpu().numpy())

        return embeddings



##
## DNABert https://doi.org/10.1093/bioinformatics/btab083
##

class DNABertEmbedder(BaseEmbedder):

    def load_model(self, 
                   dnabert_path: str = '../../external-models/DNABERT/', 
                   kmer: int = 3, ):


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
                model_input = self.tokenizer.batch_encode_plus(kmers, add_special_tokens=True, 
                                                                    max_length=512, return_tensors='pt', 
                                                                    padding=True)["input_ids"]
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


class NucleotideTransformerEmbedder(BaseEmbedder):

    def load_model(self, model_name):


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

    def load_model(self, model_path, **kwargs):

        

        # Get pretrained model
        self.model = AWDLSTMModelForInference.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

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
    def load_model(self, model_path, **kwargs):

        logging.set_verbosity_error()

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # load model        
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
    


# Class for one-hot encoding.
categories_4_letters_unknown = ['A', 'C', 'G', 'N', 'T']

class EncodeSequence:
    def __init__(self, nucleotide_categories = categories_4_letters_unknown):
        
        self.nucleotide_categories = nucleotide_categories
        
        self.label_encoder = LabelEncoder().fit(self.nucleotide_categories)
        
    
    def transform_integer(self, sequence, return_onehot = False): # integer/onehot encode sequence
        if isinstance(sequence, np.ndarray):
            return sequence
        if isinstance(sequence[0], str):  # if input is str 
            sequence = np.array(list(sequence))
        
        sequence = self.label_encoder.transform(sequence)
        
        if return_onehot:
            sequence = np.eye(len(self.nucleotide_categories))[sequence]
        return sequence
    
    def inverse_transform_integer(self, sequence):
        if isinstance(sequence, str): # if input is str
            return sequence
        sequence = EncodeSequence.reduce_last_dim(sequence) # reduce last dim
        sequence = self.label_encoder.inverse_transform(sequence)
        return ('').join(sequence)
    
    @staticmethod
    def reduce_last_dim(sequence):
        if isinstance(sequence, (str, list)): # if input is str
            return sequence
        if len(sequence.shape) > 1:
            sequence = np.argmax(sequence, axis=-1)
        return sequence

    
# backward compatibility
def embed_dnabert(sequences, path: str, kmer: int = 3, disable_tqdm = False):
    return DNABertEmbedder(path, kmer).embed(sequences, disable_tqdm = disable_tqdm)

def embed_gpn(sequences):
    return GPNEmbedder().embed(sequences)

def embed_nucleotide_transformer(sequences, model_name):
    return NucleotideTransformerEmbedder(model_name).embed(sequences)

def embed_awdlstm(sequences, model_path, disable_tqdm = False, **kwargs):
    return AWDLSTMEmbedder(model_path, **kwargs).embed(sequences, disable_tqdm = disable_tqdm )

def embed_convnet(sequences, model_path, disable_tqdm = False, **kwargs):
    return ConvNetEmbedder(model_path, **kwargs).embed(sequences, disable_tqdm = disable_tqdm)

def embed_sequence(sequences : List[str], embedding_type : str = 'categorical', **kwargs):
    '''
    sequences : list of sequences to embed
    '''
    if not embedding_type:
        return sequences
    
    if embedding_type == 'categorical' or embedding_type == 'onehot':
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
