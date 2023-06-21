# %%
import torch
from bioio.tf.utils import load_tfrecord

# %%
class TFRecordIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfrecords, **kwargs):
        super(TFRecordIterableDataset).__init__()
        self.tf_dataset = load_tfrecord(tfrecords, **kwargs)
        
    def __iter__(self):
        for example in self.tf_dataset.as_numpy_iterator():
            yield (torch.tensor(example['inputs']), torch.tensor(example['outputs']))


