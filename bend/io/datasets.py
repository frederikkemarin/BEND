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
import bioio.tf.utils as tfutils
import tensorflow as tf

def dataset_to_tfrecord(dataset, filepath, encoding='bytes'):
    features = tfutils.dataset_to_tensor_features(dataset, encoding=encoding)
    tfutils.features_to_json_file(features, filepath + '.features.json')

    with tf.io.TFRecordWriter(filepath, options=tf.io.TFRecordOptions(compression_type='ZLIB')) as tfrecord_write: # TODO : put compression type here, smth like : `options=tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.ZLIB` or `TFRecordCompressionType.ZLIB`
        for serialized_example in tfutils.serialize_dataset(dataset, features):
            tfrecord_write.write(serialized_example)

def load_tfrecord(tfrecords, features_file=None, deserialize=True, shuffle=None):
    if isinstance(tfrecords, str):
        # backward compatibility, accept a single tfrecord file instead of a list of tfrecord files
        tfrecords = [tfrecords]
    dataset = tf.data.Dataset.from_tensor_slices(tfrecords)
    #try:
    dataset = dataset.interleave(lambda fp: tf.data.TFRecordDataset(fp, compression_type='ZLIB'), cycle_length=1, block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    #except: 
    #    dataset = dataset.interleave(lambda fp: tf.data.TFRecordDataset(fp), cycle_length=1, block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle is not None:
        # shuffle examples before deserializing
        assert isinstance(shuffle, int)
        dataset = dataset.shuffle(shuffle)

    # optimize IO
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # desrialize examples using feature specification in auxiliary file
    if deserialize:
        if features_file is None:
            # if list of tfrecords is supplied but no features file, use features file of first tfrecord - this must exist
            features_file = tfrecords[0] + '.features.json'
        features = tfutils.features_from_json_file(features_file)

        dataset = dataset.map(features.deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # optimize IO
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

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


