'''
Only train the downstream model on the embeddings 
'''
import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
from  bend.utils.trainer import BaseTrainer,  MSELoss, BCEWithLogitsLoss, PoissonLoss, CrossEntropyLoss
import wandb
import os

# load config 
@hydra.main(config_path=None, config_name=None, version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
    
    # mkdir output_dir 
    os.makedirs(f'{cfg.output_dir}/checkpoints/', exist_ok=True)
    print('output_dir', cfg.output_dir)
    # init wandb
    run = wandb.init(
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project, 
        name = cfg.wandb.name, 
        mode = cfg.wandb.mode, 
        dir = cfg.output_dir, 
        config = cfg)
    
    OmegaConf.save(cfg, f"{cfg.output_dir}/config.yaml") # save the config to the experiment dir
    # set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)
    # instantiate model 
    model = hydra.utils.instantiate(cfg.model).to(device).float()
    # put model on dataparallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    print(model)

    # instantiate optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.params.lr, weight_decay = cfg.params.weight_decay) 
    
    # define criterion
    print(f'Use {cfg.params.criterion} loss function')
    if cfg.params.criterion == 'cross_entropy':
        criterion = CrossEntropyLoss(ignore_index = cfg.data.ignore_index, 
                                     weight=torch.tensor(cfg.params.class_weights).to(device) if cfg.params.class_weights is not None else None)
    elif cfg.params.criterion == 'poisson_nll':
        criterion = PoissonLoss()
    elif cfg.params.criterion == 'mse':
        criterion = MSELoss()
    elif cfg.params.criterion == 'bce':
        criterion = BCEWithLogitsLoss()

    # init dataloaders 
    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.data) # instantiate dataloaders
    print('batch size', cfg.data.batch_size)
    # instantiate trainer
    trainer = BaseTrainer(model = model, optimizer = optimizer, criterion = criterion, 
                        device = device, config = cfg, gradient_accumulation_steps = cfg.params.gradient_accumulation_steps)
    
    if cfg.params.mode == 'train':
        # train     
        trainer.train(train_loader, val_loader, cfg.params.epochs, cfg.params.load_checkpoint)

    # test 
    trainer.test(test_loader, overwrite=False)

if __name__ == '__main__':
    print('Run experiment')
    run_experiment()
