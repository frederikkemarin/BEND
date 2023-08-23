import os 

def download_model(model = 'convnet', 
                   base_url = 'https://sid.erda.dk/share_redirect/dbQM0pgSlM/pretrained_models/',
                   destination_dir : str = './pretrained_models/' # pretrained_models
                   ):
    
    """download model from url to destination directory"""
    # make destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    
    files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']
    for file in files:
        url = f'{base_url}{model}/{file}'
        os.system(f'wget {url} -P {destination_dir}/')

    return 