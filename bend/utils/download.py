"""
download.py
===========
Utility functions for downloading pretrained models from the web.
"""

import os 

def download_model(model: str = 'convnet', 
                   base_url: str = 'https://sid.erda.dk/share_redirect/dbQM0pgSlM/pretrained_models/',
                   destination_dir : str = './pretrained_models/' # pretrained_models
                   ) -> None:
    """Download BEND pretrained model checkpoints from the ERDA URL. 
    Uses wget to download the files.
    
    Parameters
    ----------
    model : str
        Model to download. Needs to be a directory name in base_url.
    base_url : str
        Base URL to download from. 
        Default is BEND's pretrained models directory on ERDA.
    destination_dir : str
        Destination directory to download to.
        Default is ./pretrained_models/

    Returns
    -------
    None.
    """
    
    # """download model from url to destination directory"""
    # make destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    
    files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']
    for file in files:
        url = f'{base_url}{model}/{file}'
        os.system(f'wget {url} -P {destination_dir}/')

    return 


def download_model_zenodo(base_url: str, destination_dir: str = './pretrained_models'):
    """
    Download a HF model hosted as a Zenodo record.
    Uses wget to download the files.
    We use this to get the GROVER model, but it should work for any model hosted on Zenodo as a flat directory.

    Parameters
    ----------
    base_url : str
        Base URL to download from.
    destination_dir : str
        Destination directory to download to.
        Default is ./pretrained_models/

    Returns
    -------
    None.
    """

    os.makedirs(destination_dir, exist_ok=True)
    # https://zenodo.org/records/8373117/files/training_args.bin?download=1
    files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
    for file in files:
        url = f'{base_url}/files/{file}'
        os.system(f'wget {url} -P {destination_dir}/')
