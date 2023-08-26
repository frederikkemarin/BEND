"""
bend.io
=======
I/O module for reading and writing data. This module provides utilities for 
processing genome coordinate-based sequence data in bed files to embeddings,
and saving and loading embedding data to and from disk in TFRecords format.

"""
from .datasets import TFRecordIterableDataset