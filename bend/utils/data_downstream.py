# create torch dataset & dataloader from tfrecord
import torch
from bend.io.datasets import TFRecordIterableDataset
from functools import partial
import os
import glob

def pad_to_longest(sequences, padding_value = -100, batch_first=True):
    
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, 
                                                padding_value=padding_value, 
                                                batch_first=batch_first)

    return sequences

def collate_fn_pad_to_longest(batch, 
                              padding_value = -100):
    
    if isinstance(batch, torch.Tensor):
        return batch

    batch = list(zip(*batch))
    padded = tuple(map(partial(pad_to_longest, padding_value = padding_value, batch_first = True), batch))

    if padding_value !=0: # make sure features do no have padding value 
        padded[0][padded[0] == padding_value] = 0

    return padded

from typing import Union

def worker_init_fn(self, _):
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

    '''Load data to dataloader from a list of paths or a single path'''
    if isinstance(data, str):
        data = [data]
    dataset = TFRecordIterableDataset(data, shuffle = shuffle)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             num_workers=num_workers, 
                                             collate_fn=partial(collate_fn_pad_to_longest, 
                                                                padding_value = padding_value))
    return dataloader

def get_data(data_dir : str, 
            train_data : list = None, 
             valid_data : list = None, 
             test_data : list = None, 
             cross_validation : Union[bool, int] = False, 
             batch_size : int = 8,
             num_workers : int = 32,
             padding_value = -100, 
             shuffle : int = None, 
             **kwargs):
    """
    Function to get data from tfrecords. 
    Args: 
        data_dir: path to data directory containing the tfrecords
        train_data: path to train tfrecord, in case of cross validation can give simply the path to the data directory
        valid_data: path to valid tfrecord
        test_data: path to test tfrecord
        cross_validation: bool/int, if int, use the given partition as test set, 
                                    +1 as valid set and the rest as train set.
                                    First split is 1.
        batch_size: int, batch size
        num_workers: int, number of workers for dataloader
        padding_value: int, value to pad sequences to longest sequence
        shuffle: int, shuffle value for the train dataloader
    """

    if cross_validation is not False:
        cross_validation = int(cross_validation) -1 
        # get basepath of data directory
        # get all tfrecords in data directory
        tfrecords = glob.glob(f'{data_dir}/*.tfrecord')
        # sort tfrecords
        #tfrecords.sort()
        tfrecords = sorted(tfrecords, key=lambda x: int(x.split('/')[-1].split('.')[0][4:]))
        test_data = tfrecords[cross_validation]
        # get valid data, cycle through tfrecords if test set is the last one
        if cross_validation == len(tfrecords) - 1:
            valid_data = tfrecords[0]
        else:
            valid_data = tfrecords[cross_validation + 1] 
        # get train data, remove test and valid data from list of tfrecords
        tfrecords.remove(test_data)
        tfrecords.remove(valid_data)
        train_data = tfrecords
    else: 
        # join data_dir with each item in train_data, valid_data and test_data 

        train_data = [f'{data_dir}/{x}' for x in train_data] if train_data else None
        valid_data = [f'{data_dir}/{x}' for x in valid_data] if valid_data else None
        test_data = [f'{data_dir}/{x}' for x in test_data] if test_data else None

    # get dataloaders
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