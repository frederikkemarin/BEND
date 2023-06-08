'''
This script trains a model on a prediction task.
'''

from bend.models import CNN, DilatedConvNet
from bend.utils import DownstreamTrainer
from bend.utils import PreembeddedDataset
import argparse




def main():
    parser = argparse.ArgumentParser('Train a model on a prediction task')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('model', choices=['cnn', 'dilatedcnn'], type=str, help='Model architecture')
    parser.add_argument('out_dir', type=str, help='Path to the output directory')
    parser.add_argument('embedding_dir', type=str, help='Path to the directory containing the embeddings')


    # TODO
    # add DownstreamTrainer and PreembeddedDataset to source
    # Work out how to control task type, stopping metrics, upsampling, hdf5 loading etc here.

