"""
data_downstream.py
==================
Data loading and processing utilities for training
downsteam tasks on embeddings saved in webdataset .tar.gz format.
"""

# create torch dataset & dataloader from webdataset
import torch
from functools import partial
import os
import glob
from typing import List, Tuple, Union
import webdataset as wds

def pad_to_longest(sequences: List[torch.Tensor], padding_value = -100, batch_first=True):
    '''Pad a list of sequences to the longest sequence in the list.
    Parameters
    ----------
    sequences : list[torch.Tensor]
        List of sequences to pad.
    padding_value : int, optional
        Value to pad with. The default is -100.
    batch_first : bool, optional
        Whether to return the batch dimension first. The default is True.
    Returns
    -------
    sequences : torch.Tensor
        Padded sequences.
    '''
    
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                padding_value=padding_value, 
                                                batch_first=batch_first)

    return sequences

def collate_fn_pad_to_longest(batch, 
                              padding_value = -100):
    '''Collate function for dataloader that pads to the longest sequence in the batch.
    Parameters
    ----------
    batch : list
        List of samples to collate.
    padding_value : int, optional
        Value to pad with. The default is -100.
    Returns
    -------
    padded : Tuple[torch.Tensor]
        Padded batch.
    '''
    
    if isinstance(batch, torch.Tensor):
        return batch

    batch = list(zip(*batch))
    padded = tuple(map(partial(pad_to_longest, padding_value = padding_value, batch_first = True), batch))

    if padding_value !=0: # make sure features do no have padding value 
        padded[0][padded[0] == padding_value] = 0

    return padded


def worker_init_fn(self, _):
    """
    Initialize worker function for data loading to make sure that each worker loads a different part of the data.
    See the pytorch data loading documentation for more information.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.data) //worker_info.num_workers
    dataset.data = dataset.data[worker_id * split_size:(worker_id +1) * split_size]



def return_dataloader(data : Union[str, list], 
                      batch_size : int = 8, 
                      num_workers : int = 0,
                      padding_value = -100, 
                      shuffle : int = None):
    """
    Function to return a dataloader from a list of tar files or a single one.
    
    Parameters
    ----------
    data : Union[str, list]
        Path to tfrecord or list of paths to tar files.
    batch_size : int, optional
        Batch size. The default is 8.
    num_workers : int, optional
        Number of workers for data loading. The default is 0.
    padding_value : int, optional
        Value to pad with. The default is -100.
    shuffle : int, optional
        Whether to shuffle the data. The default is None.
    """

    # '''Load data to dataloader from a list of paths or a single path'''
    if isinstance(data, str):
        data = [data]
    dataset = wds.WebDataset(data)
    dataset = dataset.decode() # iterator over samples - each sample is dict with keys "input.npy" and "output.npy"
    dataset = dataset.to_tuple("input.npy", "output.npy")
    dataset = dataset.map_tuple(torch.from_numpy, torch.from_numpy) # TODO any specific dtype requirements or all handled already?

    # untested from here on
    dataset = dataset.map_tuple(torch.squeeze, torch.squeeze) # necessary for collate_fn_pad_to_longest ?
    dataset = dataset.batched(batch_size, collation_fn = None) #returns list of tuples
    dataset = dataset.map(partial(collate_fn_pad_to_longest, padding_value = padding_value))


    dataloader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)

    return dataloader

def get_data(data_dir : str, 
            train_data : List[str] = None, 
             valid_data : List[str] = None, 
             test_data : List[str] = None, 
             cross_validation : Union[bool, int] = False, 
             batch_size : int = 8,
             num_workers : int = 32,
             padding_value = -100, 
             shuffle : int = None, 
             **kwargs):

    """
    Function to get data from tar files.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory containing the tar files.
    train_data : List[str], optional
        List of paths to train tar files. The default is None.
        In case of cross validation can be simply the path to the data directory.
    valid_data : List[str], optional
        List of paths to valid tar files. The default is None.
    test_data : List[str], optional
        List of paths to test tar files. The default is None.
    cross_validation : Union[bool, int], optional
        If int, use the given partition as test set, +1 as valid set and the rest as train set.
        First split is 1. The default is False.
    batch_size : int, optional
        Batch size. The default is 8.
    num_workers : int, optional
        Number of workers for data loading. The default is 0.
    padding_value : int, optional
        Value to pad with. The default is -100.
    shuffle : int, optional
        Whether to shuffle the data. The default is None.

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for training data.
    valid_dataloader : torch.utils.data.DataLoader
        Dataloader for validation data.
    test_dataloader : torch.utils.data.DataLoader
        Dataloader for test data.  
    """
    # check if data exists 
    if not os.path.exists(data_dir):
        print(data_dir)
        raise SystemExit(f'The data directory {data_dir} does not exist\nExiting script')
    if cross_validation is not False:
        cross_validation = int(cross_validation) -1 
        # get basepath of data directory
        # get all tar.gz in data directory
        tars = glob.glob(f'{data_dir}/*.tar.gz')
        # sort tar files
        tars = sorted(tars, key=lambda x: int(x.split('/')[-1].split('.')[0][4:]))
        test_data = tars[cross_validation]
        # get valid data, cycle through tar.gz if test set is the last one
        if cross_validation == len(tars) - 1:
            valid_data = tars[0]
        else:
            valid_data = tars[cross_validation + 1] 
        # get train data, remove test and valid data from list of tar files
        tars.remove(test_data)
        tars.remove(valid_data)
        train_data = tars

    # TODO chunking loading done right - need to support both this and the commented out block.
    else:
        tars = glob.glob(f'{data_dir}/*.tar.gz')
        train_data = [x for x in tars if os.path.split(x)[-1].startswith('train')]
        valid_data = [x for x in tars if os.path.split(x)[-1].startswith('valid')]
        test_data = [x for x in tars if os.path.split(x)[-1].startswith('test')]

    # else: 
    #     # join data_dir with each item in train_data, valid_data and test_data 

    #     train_data = [f'{data_dir}/{x}' for x in train_data] if train_data else None
    #     valid_data = [f'{data_dir}/{x}' for x in valid_data] if valid_data else None
    #     test_data = [f'{data_dir}/{x}' for x in test_data] if test_data else None

    # get dataloaders
    # import ipdb; ipdb.set_trace()
    train_dataloader = return_dataloader(train_data, batch_size = batch_size, 
                                         num_workers = num_workers, 
                                         padding_value=padding_value, 
                                         shuffle = shuffle) if train_data else None
    valid_dataloader = return_dataloader(valid_data, batch_size = batch_size, 
                                         num_workers = num_workers, 
                                         padding_value=padding_value, ) if valid_data else None
    test_dataloader = return_dataloader(test_data, batch_size = batch_size, 
                                        num_workers = num_workers, 
                                        padding_value=padding_value, ) if test_data else None

    return train_dataloader, valid_dataloader, test_dataloader