import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
import os
import bend.io.sequtils as sequtils
import pandas as pd
import numpy as np
import h5py
from bioio.tf import dataset_from_iterable
#from bioio.tf import dataset_to_tfrecord
from bend.io.datasets import dataset_to_tfrecord
import sys
# load config 
@hydra.main(config_path="../conf/embedding/", config_name="embed", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a embedding of nucleotide sequences.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    print('Embedding data for', cfg.task)
    # read the bed file and get the splits :  
    if not 'splits' in cfg or cfg.splits is None:
        splits = sequtils.get_splits(cfg[cfg.task].bed) 
    else:
        splits = cfg.splits
    print('Embedding with', cfg.model) 
    # instatiante model
    embedder = hydra.utils.instantiate(cfg[cfg.model])
    for split in splits:
        print(f'Embedding {split} set')
        output_dir = f'{cfg.data_dir}/{cfg.task}/{cfg.model}/'
        
        os.makedirs(output_dir, exist_ok=True)

        # embed in chunks 
        # get length of bed file and divide by chunk size, if a spcific chunk is not set 
        df = pd.read_csv(cfg[cfg.task].bed, sep = '\t', low_memory=False)
        df = df[df.iloc[:, -1] == split] if split is not None else df
        possible_chunks = list(range(int(len(df) /cfg.chunk_size)+1))         
        if cfg.chunk is None: 
            cfg.chunk = possible_chunks
        # embed in chunks
        for n, chunk in enumerate(cfg.chunk): 
            if os.path.exists(f'{output_dir}/{split}_{chunk}.hdf5'):
                return
            # create hdf5 file to write too
            ds = h5py.File(f'{output_dir}/{split}_{chunk}.hdf5', 'w')
            dims = (len(df), 
                    cfg.datadims[f'{cfg.task}_length'], 
                    cfg.datadims[cfg.model] ) if cfg.model != 'onehot' else (len(df), 
                                                                            cfg.datadims[f'{cfg.task}_length'] )
            ds.create_dataset('inputs', (dims), 
                            compression="gzip",
                            dtype='float64')
            
            dims = (len(df), cfg.datadims[f'{cfg.task}_length']) if cfg.datadims.output_downsample_window[cfg.task] is None else (len(df), cfg.datadims[f'{cfg.task}'])
            ds.create_dataset('labels', (dims), 
                            compression="gzip",
                            dtype='float64')
            ds.close()

            print(f'Embedding chunk {chunk}/{len(possible_chunks)}')
            sequtils.embed_from_bed(**cfg[cfg.task], embedder = embedder, 
                                        output_path = f'{output_dir}/{split}_{chunk}.hdf5',
                                        split = split, chunk = chunk, chunk_size = cfg.chunk_size,   
                                        upsample_embeddings = cfg[cfg.model]['upsample_embeddings'] if 'upsample_embeddings' in cfg[cfg.model] else False)
            
            
        

if __name__ == '__main__':
    
    print('Run Embedding')
    
    run_experiment()