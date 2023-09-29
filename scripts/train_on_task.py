'''
train_on_task.py
----------------
Train a model on a downstream task.
'''
import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
from  bend.utils.task_trainer import BaseTrainer,  MSELoss, BCEWithLogitsLoss, PoissonLoss, CrossEntropyLoss
import wandb
from bend.models.downstream import CustomDataParallel
import os
from hydra.core.hydra_config import HydraConfig
from bend.models import downstream
# load config 
@hydra.main(config_path=f"../conf/supervised_tasks/", config_name=None ,version_base=None) #
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a supervised task experiment.
    This function is called by hydra.
    
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
    # mkdir output_dir 
    os.makedirs(f'{cfg.output_dir}/checkpoints/', exist_ok=True)
    print('output_dir', cfg.output_dir)
    # exit script if best model is already tested 
    if os.path.exists(f'{cfg.output_dir}/best_model_metqrics.csv'):
        return 'Best model already tested'
    # init wandb
    run = wandb.init(**cfg.wandb, dir = cfg.output_dir, config = cfg)
    
    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml") # save the config to the experiment dir
    # set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate model 
    model = hydra.utils.instantiate(cfg.model, 
                                    encoder = cfg.misc.resnet_encoder if cfg.embedder == 'resnet-supervised' else None)    
    # put model on dataparallel
    if torch.cuda.device_count() > 1:
        if cfg.params.parallel is True: 
            from bend.models.downstream import CustomDataParallel
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = CustomDataParallel(model)
        else: 
            # set gpu according to job number from sweep
            device_to_use = HydraConfig.get().job.num % torch.cuda.device_count()
            device = torch.device(f'cuda:{torch.cuda.device(device_to_use).idx}')
            print(f'Use gpu {torch.cuda.get_device_name(device=device)}')
            
    model = model.to(device).float() 
    print(model)

    # instantiate optimizer
    optimizer =  hydra.utils.instantiate(cfg.optimizer, params = model.parameters())
    
    # define criterion
    print(f'Use {cfg.params.criterion} loss function')
    if cfg.params.criterion == 'cross_entropy':
        criterion = CrossEntropyLoss(ignore_index = cfg.data.padding_value, 
                                     weight=torch.tensor(cfg.params.class_weights).to(device) if cfg.params.class_weights is not None else None)
    elif cfg.params.criterion == 'poisson_nll':
        criterion = PoissonLoss()
    elif cfg.params.criterion == 'mse':
        criterion = MSELoss()
    elif cfg.params.criterion == 'bce':
        criterion = BCEWithLogitsLoss(class_weights=torch.tensor(cfg.params.class_weights).to(device) if cfg.params.class_weights is not None else None)

    # init dataloaders 
    if 'supervised' in cfg.embedder : cfg.data.data_dir = cfg.data.data_dir.replace(cfg.embedder, 'onehot')
    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.data) # instantiate dataloaders
    # instantiate trainer
    trainer = BaseTrainer(model = model, optimizer = optimizer, criterion = criterion, 
                        device = device, config = cfg)
    
    if cfg.params.mode == 'train':
        # train     
        trainer.train(train_loader, val_loader, cfg.params.epochs, cfg.params.load_checkpoint)

    # test 
    trainer.test(test_loader, overwrite=False)

    return 

if __name__ == '__main__':
    print('Run experiment')
    run_experiment()
