import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
import os
import bend.io.sequtils as sequtils
import pandas as pd
from bioio.tf import dataset_from_iterable
from bioio.tf import dataset_to_tfrecord

# load config 
@hydra.main(config_path="../conf/embedding/", config_name="embed", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    for task in cfg.tasks:
        # read the bed file and get the splits : 
        splits = sequtils.get_splits(cfg[task].bed)
        for split in splits:
            for model in cfg.models: 
                output_dir = f'{cfg.data_dir}/{task}/{model}/'
                os.makedirs(output_dir, exist_ok=True)
                # instatiante model
                embedder = hydra.utils.instantiate(cfg[model])
                gen = sequtils.embed_from_bed(**cfg[task], embedder = embedder, split = split)
                # save the embeddings to tfrecords 
                dataset = dataset_from_iterable(gen)
                dataset.element_spec
                dataset_to_tfrecord(dataset, f'{output_dir}/{split}.tfrecord')




if __name__ == '__main__':
    print('Run experiment')
    run_experiment()