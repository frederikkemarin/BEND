'''
Only train the downstream model on the embeddings 
'''
import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
from  bend.utils.task_trainer import BaseTrainer,  MSELoss, BCEWithLogitsLoss, PoissonLoss, CrossEntropyLoss
import wandb
from bend.models.downstream import CustomDataParallel
import os
import sys

# load config 
@hydra.main(config_path=f"../conf/supervised_tasks/", config_name=None ,version_base=None) #
def run_experiment(cfg: DictConfig) -> None:
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
    # mkdir output_dir 
    os.makedirs(f'{cfg.output_dir}/checkpoints/', exist_ok=True)
    print('output_dir', cfg.output_dir)
    # init wandb
    run = wandb.init(**cfg.wandb, dir = cfg.output_dir, config = cfg)
    
    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml") # save the config to the experiment dir
    # set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    # instantiate model 
    model = hydra.utils.instantiate(cfg.model).to(device).float()
    # put model on dataparallel
    if torch.cuda.device_count() > 1:
        from bend.models.downstream import CustomDataParallel
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = CustomDataParallel(model)
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
        criterion = BCEWithLogitsLoss()

    # init dataloaders 
    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.data) # instantiate dataloaders
    # instantiate trainer
    trainer = BaseTrainer(model = model, optimizer = optimizer, criterion = criterion, 
                        device = device, config = cfg)
    
    if cfg.params.mode == 'train':
        # train     
        trainer.train(train_loader, val_loader, cfg.params.epochs, cfg.params.load_checkpoint)

    # test 
    trainer.test(test_loader, overwrite=False)

if __name__ == '__main__':
    print('Run experiment')
    run_experiment()
