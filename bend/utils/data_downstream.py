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

def get_data(train_data : Union[str, list], 
             valid_data : Union[str, list] = None, 
             test_data : Union[str, list] = None, 
             cross_validation : Union[bool, int] = False, 
             batch_size : int = 8,
             num_workers : int = 0,
             padding_value = -100, 
             shuffle : int = None):
    """
    Function to get data from tfrecords. 
    Args: 
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
        basepath = os.path.dirname(train_data)
        # get all tfrecords in data directory
        tfrecords = glob.glob(os.path.join(basepath, '*.tfrecord'))
        # sort tfrecords
        tfrecords.sort()
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
    # get dataloaders
    train_dataloader = return_dataloader(train_data, batch_size = batch_size, 
                                         num_workers = num_workers, 
                                         padding_value=padding_value, 
                                         shuffle = shuffle)
    valid_dataloader = return_dataloader(valid_data, batch_size = batch_size, 
                                         num_workers = num_workers, 
                                         padding_value=padding_value, )
    test_dataloader = return_dataloader(test_data, batch_size = batch_size, 
                                        num_workers = num_workers, 
                                        padding_value=padding_value, )

    return train_dataloader, valid_dataloader, test_dataloader