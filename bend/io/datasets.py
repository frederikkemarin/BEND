"""
datasets.py
===========
Dataset classes for loading data from tfrecord files.
Uses https://github.com/mhorlacher/bioio for loading tfrecords.
The code here only handles iteration and conversion to torch tensors.
"""
# %%
import torch
from bioio.tf.utils import load_tfrecord

# %%
class TFRecordIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset for loading data from tfrecords.
    """
    def __init__(self, tfrecords, **kwargs):
        super(TFRecordIterableDataset).__init__()
        self.tf_dataset = load_tfrecord(tfrecords, **kwargs)
        
    def __iter__(self):
        """
        Iterate over the tfrecord dataset and yield a tuple of torch tensors.
        
        Yields
        ------
        Tuple[torch.Tensor]
            Tuple of 2 torch tensors, one for inputs and one for outputs.
        """
        for example in self.tf_dataset.as_numpy_iterator():
            yield (torch.tensor(example['inputs']), torch.tensor(example['outputs']))


