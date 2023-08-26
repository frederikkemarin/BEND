import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
import os
import bend.io.sequtils as sequtils
import pandas as pd
from bioio.tf import dataset_from_iterable
from bioio.tf import dataset_to_tfrecord
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
    for task in cfg.tasks:
        print('Embedding data for', task)
        # read the bed file and get the splits :  
        if not 'splits' in cfg or cfg.splits is None:
            splits = sequtils.get_splits(cfg[task].bed) 
        else:
            splits = cfg.splits
        for model in cfg.models:
            print('Embedding with', model) 
            # instatiante model
            embedder = hydra.utils.instantiate(cfg[model])
            for split in splits:
                print(f'Embedding {split} set')
                output_dir = f'{cfg.data_dir}/{task}/{model}/'
                os.makedirs(output_dir, exist_ok=True)
                gen = sequtils.embed_from_bed(**cfg[task], embedder = embedder, split = split, 
                                             upsample_embeddings = cfg[model]['upsample_embeddings'] if 'upsample_embeddings' in cfg[model] else False)
                # save the embeddings to tfrecords 
                dataset = dataset_from_iterable(gen)
                dataset.element_spec
                dataset_to_tfrecord(dataset, f'{output_dir}/{split}.tfrecord')




if __name__ == '__main__':
    
    print('Run Embedding')
    
    run_experiment()