import torch 
import wandb
import sys
import os
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, average_precision_score
from sklearn.feature_selection import r_regression
import sys 
sys.path.append("..")
from utils import make_embeddings
import pandas as pd
from typing import Union


# make a base trainer class for training

class BaseTrainer:
    ''''Perform training and validation steps for a given model and dataset'''

    def __init__(self, 
                model, 
                optimizer, 
                criterion, 
                device, 
                config, 
                overwrite_dir=False, 
                gradient_accumulation_steps: int = 1, ):


        # add metric under config.params
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.overwrite_dir = overwrite_dir
        self._create_output_dir(self.config.output_dir) # create the output dir for the model 
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler() # init scaler for mixed precision training
        
    
    def _create_output_dir(self, path):
        os.makedirs(f'{path}/checkpoints/', exist_ok=True)
        if self.overwrite_dir or not os.path.exists(f'{path}/losses.csv'):
            pd.DataFrame(columns = ['Epoch', 'train_loss', 'val_loss', f'val_{self.config.params.metric}']).to_csv(f'{path}/losses.csv')
        return 
    
    def _load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        val_acc = checkpoint['val_acc']
        val_metric = checkpoint[f'val_{self.config.params.metric}']
        return epoch, train_loss, val_loss, val_metric
    
    def _save_checkpoint(self, epoch, train_loss, val_loss, val_acc, val_mcc):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            f'val_{self.config.params.metric}': val_metric
            }, f'{self.config.output_dir}/checkpoints/epoch_{epoch}.pt')
        return
    
    def _log_loss(self, epoch, train_loss, val_loss, val_metric):
        df = pd.read_csv(f'{self.config.output_dir}/losses.csv')
        df = pd.concat([df, pd.DataFrame([[epoch, train_loss, val_loss, val_metric]], 
                                         columns = ['Epoch', 'train_loss', 'val_loss', f'val_{self.config.params.metric}'])
                                         ], ignore_index=True)
        df.to_csv(f'{self.config.output_dir}/losses.csv', index = False)
        return
    
    def _log_wandb(self, epoch, train_loss, val_loss, val_acc, val_mcc):
        wandb.log({'train_loss': train_loss, 
                   'val_loss': val_loss, 
                   f'val_{self.config.params.metric}': val_metric}, 
                   step = epoch)
        
        # wandb.log({"Training latent with labels": wandb.Image(plt)})
        return