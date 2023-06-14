from torch.utils.data import Sampler
import random
import torch
import numpy as np
from operator import itemgetter
from torch.utils.data import Dataset
from functools import partial
from typing import List, Optional, Tuple, Union
import pandas as pd
import omegaconf
from Bio.Seq import Seq
import glob
try:
    from graph_part import train_test_validation_split
except: 
    pass
import os 
import random
from collections import Counter
import sys
from .sequences import EncodeSequence
from .make_embeddings import embed_sequence
from operator import itemgetter




def to_list(arr):
    return(arr.tolist())


encode_seq = EncodeSequence()
def choose_strand(seq, label =None, random_seed = 42, reverse_complement = True):

    '''Randomly sample the direction of the dna segment'''    
    
    #random_generator = random_generator #np.random.default_rng(seed = random_seed) if random_generator is None else random_generator
    strand = np.random.choice(2)#(["+", "-"])
    #strand = random_generator.choice(["+", "-"])
    seq = encode_seq.inverse_transform_integer(seq)
    if strand == "0":
        seq = str(Seq(seq).reverse_complement())
        label = torch.flip(label, dims= (0,))

    return seq, label


class ListDataset(Dataset):
    def __init__(self, 
                data : Union[List[torch.tensor], List[str], str], 
                indices : List[int] = None,
                sample_strand = False, 
                seed = 42,
                embedding = None, 
                keys : List[str] = ['features_onehot', 'labels_simple_direction_DA', 'mask'], 
                **kwargs):
        """
        Args:
            data (list): (mulitple) lists each containing torch.tensors
        """
        self.data_path = data
        self.indices = indices
        self.kwargs = kwargs
        
        if isinstance(data, str): # then the data is the path to the dataset
            dataset = glob.glob(f'{data}/*torch')
            # sort files numerically 
            dataset.sort(key=lambda f: int(os.path.basename(f).split('.')[0]))
            if indices is not None:
                #dataset = itemgetter(*indices)(dataset)
                dataset = tuple([item for item in dataset if int(os.path.basename(item).split('.')[0]) in indices])
            self.dataset = dataset
        elif isinstance(data[0], str): # then the dataset is given as a list of paths to torch files
            self.dataset = data
        elif isinstance(data[0], torch.Tensor):
            self.dataset = list(zip(*data))
        self.sample_strand = sample_strand
        self.embedding = embedding
        self.keys = keys

    def get_lengths(self):
        # load csv from data_path
        # check if files exists
        if os.path.exists(f'{self.data_path}/metadata.csv'):
            lengths = pd.read_csv(f'{self.data_path}/metadata.csv')[['index','length']]
            # if no indices given then return whole file 
            # else get the lengths of the files at the indices
            if self.indices is not None:
                lengths = lengths.iloc[self.indices]
                lengths['index'] = [i for i in range(len(lengths))]
            # get values as list of tuples
            lengths = list(lengths.itertuples(index=False, name=None))
        else:
            lengths = [(i, len(item[0])) for i, item in enumerate(self.dataset)]

        return lengths
        

    def __getitem__(self, index):
        # if dataset is just a list of indices then load the example based on indices 
        item =  self.dataset[index]
        if isinstance(item, str): # then load the torch file
            item = torch.load(item)
            item = [item[key] for key in self.keys]
        else: 
            item = list(self.dataset[index])

        if self.sample_strand:
            try:
                item[0], item[1] = choose_strand(item[0], label = item[1])
            except: 
                item[0], _ = choose_strand(item[0])
        if self.embedding:
            item[0] = embed_sequence([item[0]], **self.embedding)[0]
        return tuple(item)#, self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def __remove__(self):
        pass



def pad_length_right_end(tensor, n_pad, value=0):
    if not torch.is_tensor(tensor):
        return tensor
    pad = [0 for i in range(len(tensor.shape)*2)]
    pad[-1] = n_pad
    
    return torch.nn.functional.pad(tensor, pad, 'constant', value)

def pad_to_longest(sequences, padding_value = -100, batch_first=True):
    try: 
        if not isinstance(sequences[0], torch.Tensor):
            try:
                sequences = [torch.from_numpy(s) for s in sequences]
            except:
                sequences = [torch.tensor(s) for s in sequences]
        sequences = [s[0] if s.dim() > 2 else s for s in sequences]
        sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                    padding_value=padding_value, batch_first=batch_first)
    except:
        sequences = sequences
    return sequences

def make_tensors(sequences):
    try:
        sequences = [torch.from_numpy(s) for s in sequences]
    except:
        sequences = [torch.tensor(s) if not isinstance(s, torch.Tensor)  else s for s in sequences]
    
    if sequences[0].dim() == 3:
        sequences = [s[0] if s.dim() > 2 else s for s in sequences]
        sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                    padding_value=0, 
                                                    batch_first=True)
        return sequences#torch.cat(sequences, dim=0)

    return torch.stack(sequences)

def collate_fn_make_tensors(batch : List[Tuple]):
    batch = list(zip(*batch))
    batch = tuple(map(make_tensors, batch))
    return batch

def collate_fn_pad_to_longest(batch : List[tuple], 
                              padding_value = -100):
    # if there is 4 elements in batch tuples (i.e. cls token is included) then do not pad on cls token 

    batch = list(zip(*batch))
    if len(batch) > 3:
        cls_tokens = batch[-1]
        cls_tokens = torch.stack([torch.tensor(item).mean(0) for item in cls_tokens]) # avg cls tokens and stack to a batch 
        batch = batch[:-1]
    padded = tuple(map(partial(pad_to_longest, padding_value = padding_value, batch_first = True), batch))

    if padding_value !=0:
        try: 
            padded[0][padded[0] == padding_value] = 0
        except:
            pass
    if 'cls_tokens' in locals(): # add the cls tokens to the padded batch 
        padded = padded + (cls_tokens,)
        
    return padded

class BatchSamplerSimilarLength(Sampler):
    """
    Batch sampler that samples indices of a dataset with simlar lengths
    
    Attributes: 
        batch_size (int) : size of resulting batches 
        super_batch_size (int) : size of the superset of indices that will be shuffled together 
        shuffle (bool) : Whether to shuffle batches 
        indices (List[Tuple[int, int]]) : Contains the the (index, length) of each datapoint in the dataset
        
        
    """
    def __init__(self, 
                 dataset : ListDataset, 
                 batch_size : int = 32, 
                 indices : List = None, 
                 sample_by_length : bool = True,
                 shuffle : bool = True, 
                 superset_factor : int = 4, 
                 padding_value = -100):
        
        """
        Args: 
            dataset (ListDataset) : a dataset containing a List[Tuple[*torch.tensor]]
            batch_size (int) : batch size 
            super_batch_size (int) : size of the superset of indices that will respectively be shuffled  
            shuffle (bool) : Whether to shuffle batches 
            indices (List[Tuple[int, int]]) : Contains the the (index, length) of each datapoint in the dataset
        """
        self.batch_size = batch_size
        self.super_batch_size = superset_factor * batch_size
        self.shuffle = shuffle
        self.sample_by_length = sample_by_length
        if self.sample_by_length is True:
            # get the indicies and length
            self.indices = dataset.get_lengths() #[(i, len(item[0])) for i, item in enumerate(dataset)]
        else: 
            # make a set of indices with all the same length
            self.indices = [(i, 1) for i in range(len(dataset))]
            self.super_batch_size = len(dataset)

        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

    def to_list(self, arr):
        return(arr.tolist())
    
    def __iter__(self):
        """
        Returns indices of the dataset batched by similar length. 
    
        Sorts indices by data length, divides them into a set of super-batches. 
        Shuffles each of the super-batches and then pends them together. 
        Then divides the indices into batches. 
        Because of this the length of the datapoints in the will be similar but each batch will contain some randomness
        """
        if self.shuffle:
            random.shuffle(self.indices)
        
        # split indices into super set of pooled by length 
        superset = tuple(np.array_split(np.array(sorted(self.indices, key=itemgetter(1))), 
                                        (len(self.indices) // self.super_batch_size)))# + 1 ))
        # shuffle each of the supersets with map
        if self.shuffle:
            _ = list(map(np.random.shuffle, superset))
        # make one list again 
        pooled_indices = np.concatenate([i[:, 0] for i in superset])
        # get batches
        batches = torch.split(torch.tensor(pooled_indices), self.batch_size) # torch.split(torch.tensor(tst), 7)
        # shuffle batchs 
        batches = list(map(self.to_list, batches))
        # shuffle batches 
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            # chose strand in batch 
            yield batch

    def __len__(self):
        return len(self.indices) // self.batch_size



    
def load_data(*filepaths, keys : list = ['features_onehot', 'labels_simple_direction_DA', 'mask']):
    '''
    Take file paths
    Return list of dictionaries
    '''
    data = []
    check_keys = ["train", "val", "test"]
    for file in filepaths:
        if isinstance(file, str):
            file = torch.load(file)
        if set(check_keys) <= file.keys(): # if the dataset is already split 
            for item in check_keys:
                data.append([file[item].get(key) for key in keys])
        
        else:
            data.append([file.get(key) for key in keys])
    return data

def return_torch_dataset(data):
    '''
    input dataset (lists or tensors) and return torch Dataset 
    '''
    # load training set
    if isinstance(data[0], list):
        dataset = ListDataset(*data)
    else:    
        dataset = torch.utils.data.TensorDataset(data)
    
    return dataset

def return_torch_dataloader(dataset, 
                            batch_size = 64, 
                            collate_fn = None,
                            superset_factor=4, 
                            shuffle=True, 
                            num_workers = 16, 
                            sample_strand = False, 
                            padding_value = -100, 
                            sample_by_length = True):
    if isinstance(dataset.dataset, ListDataset) or isinstance(dataset, ListDataset):
        dataset.sample_strand = sample_strand
        
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                #collate_fn=partial(collate_fn_pad_to_longest, padding_value = padding_value), 
                                                collate_fn = collate_fn,
                                                num_workers = num_workers, 
                                                batch_sampler = BatchSamplerSimilarLength(dataset, 
                                                                                          batch_size = batch_size, 
                                                                                          superset_factor = superset_factor, 
                                                                                          shuffle = shuffle, 
                                                                                          sample_by_length = sample_by_length), 
                                                                                          pin_memory = True)
    #else: 
    #    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)
    
    return dataloader
        

def random_split_dataset(dataset, val_size = 0.1, test_size = 0.1, seed=42, same_set = False):
    if seed:
        torch.manual_seed(seed)
    
    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)
    
    

    # this function will automatically randomly split  dataset 

    if same_set is True: 
        dataset = dataset, dataset, dataset
        
        
        
    else:
        dataset = torch.utils.data.random_split(dataset, [
                    (dataset.__len__() - (test_amount + val_amount)), 
                    val_amount, 
                    test_amount])
    
    torch.manual_seed(torch.initial_seed())
    return dataset

def split_custom(data, split_on='chromosome', val_size = 0.1, test_size = 0.1, noise = 0.01, seed=42):
    random.seed(seed)
    sizes = sorted([('val', val_size), ('test', test_size)], key=lambda x: x[1])
    if isinstance(data, str):
        data = torch.load(data)
    keys = list(Counter(data[split_on]).keys())
    values = np.array(list(Counter(data[split_on]).values())) 
    
    # zip key and percent size of total 
    zipped = list(zip(keys, values / sum(values)))
    # sort dict by ascending 
    all_keys = dict(sorted(zipped, key=lambda x: x[1]))
    partitions = {'train': [], 'val': [], 'test': []}  
    
    
    # retrieve the smallest sets  
    keep_keys = [k for k, v in all_keys.items() if v < sizes[0][1] + noise]
    
    size = 0
    # smalles partition first 
    while size < sizes[0][1] - noise:
        random.shuffle(keep_keys)
        # sample a key
        k = keep_keys.pop() #random.sample(keep_keys, 1)[0]
        # add it to size
        size += all_keys[k]
        # add key to partition
        partitions[sizes[0][0]].append(k)
        # delete key from original dictionary 
        del all_keys[k]
    
    keep_keys = [k for k, v in all_keys.items() if v < sizes[1][1] + noise]
    while sum([all_keys[k] for k in keep_keys]) < sizes[1][1]:
        try:
            noise *=2
            keep_keys = [k for k, v in all_keys.items() if v < sizes[1][1] + noise]
            
        except:    
            pass

    size = 0
    while size < sizes[1][1] - noise:
        random.shuffle(keep_keys)
        # sample a key
        k = keep_keys.pop() #random.sample(keep_keys, 1)[0]
        # add it to size
        size += all_keys[k]
        # add key to partition
        partitions[sizes[1][0]].append(k)
        # delete key from original dictionary 
        del all_keys[k]
    partitions['train'] = all_keys.keys()
    new_split = {'train' : {}, 'val' : {}, 'test' : {}}
    
    # now get the indices for each 
    
    
    for part in partitions.keys():
        # get the indexes for the partition
        part_idx = [n for n, item in enumerate(data[split_on]) if item in partitions[part]]
        # get the values of that partition
        for k, v in data.items():
            new_split[part][k] = [item for n, item in enumerate(v) if n in part_idx]
    
    random.seed()
    return new_split 


def graph_part_split_indices_from_precomputed(edge_file : str, threshold : float = 0.8, valid_size : float = 0.1, test_size : float = 0.1):

    # create dummy seqs
    edges = pd.read_csv(edge_file, header = None, index_col=None)
    pairs = edges.iloc[:, [0, 1]].values.tolist()
    flat_list = [item for sublist in pairs for item in sublist]
    flat_list.sort(key = lambda f : int(f.split('_')[-1]))
    n_seqs = int(flat_list[-1].split('_')[-1])
    
    sequences = ['A' for i in range(n_seqs)]

    # get splits

    train_idx, test_idx, valid_idx = train_test_validation_split(sequences = sequences,
                                                                edge_file = edge_file, 
                                                                alignment_mode= 'precomputed', 
                                                                threads = 8,
                                                                threshold = 0.8,
                                                                test_size = test_size,
                                                                valid_size = valid_size, 
                                                                metric_column=2)
    return train_idx, valid_idx, test_idx
    #return {'train' : train_idx, 'val' : valid_idx, 'test' : test_idx}


def split_data_from_indices(data : dict[list], indices : List[List[int]], remove_col : str):
    if indices is None:
        return data
    new_split = {}
    for k, v in data.items():
        new_split['train'][k] = [item for n, item in enumerate(v) if n in indices[0] and k != remove_col]
        new_split['val'][k] = [item for n, item in enumerate(v) if n in indices[1] and k != remove_col] 
        new_split['test'][k] = [item for n, item in enumerate(v) if n in indices[2] and k != remove_col]

    return new_split

        


def split_data(data, 
               keys : list = ['features_onehot', 'labels_simple_direction_DA', 'mask'], 
               return_ : str = 'dataloader', 
               split : str = 'random', 
               split_on = 'chromosome',
               graphpart_threshold : float = 0.8,
               seed : int = 42,
               batch_size : int = 64, 
               same_set : bool = False,
               shuffle_train_set : bool = True,
               val_size : float = 0.1, test_size : float = 0.1, 
               superset_factor : int = 4, 
               edge_file : str = None, # precomputed edgefile for graphpartitioning
               protein_col = 'protein', # column/key containing protein sequences
               num_workers = 32,
               sample_strand = False, 
               indices : Union[tuple[List], str]= None, 
               embedding = None,
               padding_value = -100, 
               sample_by_length : bool = True,
               **kwargs, 
              ):
    '''
    Take list of dictionaries.\
    Returns: 
        If list is longer than 
    split : ("random", "graphpart")
    '''

    # check if data ends with .torch
    if isinstance(data, str):
        if data.endswith('.torch'):
            data = torch.load(data)
        else:
            try:
                # data is a directory, load metadata:
                metadata = pd.read_csv(f'{data}/metadata.csv')
            except:
                print('no metadata available')
    
    if indices is not None:
        if isinstance(indices, str):
            # read a pandas dataframe with indices
            indices = pd.read_csv(indices, sep = '\t', index_col=0)
            train_indices = indices[indices['split'] == 'train'].index.tolist()
            val_indices = indices[indices['split'] == 'valid'].index.tolist()
            test_indices = indices[indices['split'] == 'test'].index.tolist()
            indices = [train_indices, val_indices, test_indices]
        split = 'Splitting with indices from file'
    # if same set then do not split 
    if same_set is True or split=='none' or split == 'same':
        same_set = True
        split = 'none'
        indices = [None]
        # do not split at all
        pass
    
    # if split is graphpart then get indices from a precomputed edgefile
    elif split == 'graphpart':
        # get indices for graphpart split 
        indices = graph_part_split_indices_from_precomputed(edge_file, threshold = graphpart_threshold, valid_size = val_size, test_size = test_size)
    elif split == 'random':
        # shuffle indices from metadata 
        indices = metadata.sample(frac=1)['index'].tolist()
        # split into train, val, test indices
        indices = indices[:int(len(indices)*(1-val_size-test_size))], indices[int(len(indices)*(1-val_size-test_size)):int(len(indices)*(1-test_size))],  
    elif split == 'presplit':
        print('Data already split')
        data = [data]

    print('Split of dataset:', split)
    # TODO : repimplement this : all splitting methods  should return only indices
    #elif split == 'chromosome':
    #    data = split_custom(data, split_on=split_on, val_size=val_size, test_size=test_size, seed=seed)
    
    # if data loaded then split data
    if not isinstance(data, str) and split != 'presplit':
        data = split_data_from_indices(data, indices, remove_col = protein_col)
    
    # TODO : reimplement this, instead of this below, then if split = 'random' then get indices for random split instead
    #if len(data) == 1: # if the data is not split in advance
    #    if split =='random' or split == 'none':
    #        data = random_split_dataset(data[0], val_size=val_size, test_size=test_size, seed=seed, same_set=same_set)
    # get the datasets 
    # save indices as dict to file 
    torch.save({'train': indices[0], 'val': indices[1], 'test': indices[2]}, f'/z/home/frma/bend_gene_annotation/indices.torch')
    datasets = []
    for n, item in enumerate(indices):
        # TODO : check id this works based on pre loaded data (so path to a torch file is given)
        datasets.append(ListDataset(data, indices = item, keys = keys, seed = seed, embedding = embedding, sample_strand = sample_strand)) 
        # no need to add strand sampling, that is done during the dataloader creation 
    if return_ == 'dataset':
        return datasets

    #if return_ == 'dataloader':
    dataloaders = []
    if sample_by_length is True:
        print('Sampling by length, pad to longest in batch')
        collate_function  = partial(collate_fn_pad_to_longest, padding_value = padding_value)
    else:
        collate_function = collate_fn_make_tensors

    dataloaders.append(return_torch_dataloader(datasets[0], batch_size=batch_size, 
                                               collate_fn=collate_function,
                                                superset_factor=superset_factor, 
                                                num_workers=num_workers, 
                                                sample_strand = sample_strand, 
                                                sample_by_length = sample_by_length,
                                                shuffle = shuffle_train_set, 
                                                padding_value = padding_value) )
    dataloaders.append(return_torch_dataloader(datasets[1], batch_size=batch_size, 
                                               collate_fn=collate_function,
                                                superset_factor=superset_factor, 
                                                shuffle=False, 
                                                sample_by_length = sample_by_length,
                                                num_workers=num_workers, 
                                                padding_value = padding_value) )
    if len(datasets) == 3: 
        dataloaders.append(return_torch_dataloader(datasets[2], batch_size=batch_size,
                                                collate_fn=collate_function, 
                                                superset_factor=superset_factor, 
                                                sample_by_length = sample_by_length,
                                                shuffle=False, 
                                                num_workers=num_workers, padding_value = padding_value) )
    else: 
        dataloaders.append(None)
    
    return dataloaders
            