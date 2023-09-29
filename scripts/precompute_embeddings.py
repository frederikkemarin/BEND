import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
import os
import bend.io.sequtils as sequtils
import pandas as pd
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
        if cfg.chunk is None: 
            cfg.chunk = list(range(int(len(pd.read_csv(cfg[cfg.task].bed, sep = '\t')) /cfg.chunk_size)))
            
        # embed in chunks
        for chunk in cfg.chunk: 
            print(f'Embedding chunk {chunk}')
            gen = sequtils.embed_from_bed(**cfg[cfg.task], embedder = embedder, split = split, chunk = chunk, chunk_size = cfg.chunk_size,   
                                        upsample_embeddings = cfg[cfg.model]['upsample_embeddings'] if 'upsample_embeddings' in cfg[cfg.model] else False)
            # save the embeddings to tfrecords 
            dataset = dataset_from_iterable(gen)
            dataset.element_spec
            dataset_to_tfrecord(dataset, f'{output_dir}/{split}_{chunk}.tfrecord')
        



if __name__ == '__main__':
    
    print('Run Embedding')
    
    run_experiment()