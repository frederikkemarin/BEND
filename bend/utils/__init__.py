"""
bend.utils
==========
This module contains a collection of utilities used throughout the project for 
data processing, model training, and evaluation.

- :class:`~bend.utils.retrieve_from_bed.Annotation`: a class for retrieving
    sequences from a reference genome based on a bed file.

- :class:`~bend.utils.task_trainer.TaskTrainer`: a class for training a model
    on a given task.
    

"""

from .retrieve_from_bed import Annotation
from .data_downstream import get_data
