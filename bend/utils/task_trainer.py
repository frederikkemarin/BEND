"""
task_trainer.py
===============
Trainer class for training downstream models on supervised tasks
"""
import torch 
import torch.nn as nn
import wandb
import os
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.feature_selection import r_regression
import pandas as pd
from typing import Union
import numpy as np

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss for classification tasks. Wrapper around `torch.nn.CrossEntropyLoss`
    that takes care of the dimensionality of the input and target tensors.
    """
    def __init__(self, 
                 ignore_index = -100, 
                 weight = None):
        """
        Get a CrossEntropyLoss object that can be used to train a model.

        Parameters
        ----------
        ignore_index : int, optional
            Index to ignore in the loss calculation. 
            Passed to `torch.nn.CrossEntropyLoss`. The default is -100.
        weight : torch.Tensor, optional
            Weights to apply to the loss. Passed to `torch.nn.CrossEntropyLoss`.
            The default is None.
        """
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, pred, target):
        """
        Calculate the cross entropy loss for a given prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor of logits.
        target : torch.Tensor
            Target tensor of labels.

        Returns
        -------
        loss : torch.Tensor
            Cross entropy loss.
        """
        criterion = torch.nn.CrossEntropyLoss(ignore_index = self.ignore_index, 
                                              weight=self.weight)
        return criterion(pred.permute(0, 2, 1), target)

class PoissonLoss(nn.Module):
    """
    Poisson loss for regression tasks.
    """
    def __init__(self):
        """
        Get a PoissonLoss object that can be used to train a model.
        """
        super(PoissonLoss, self).__init__()
    
    def _log(self, t, eps = 1e-20):
        return torch.log(t.clamp(min = eps))
    
    def _poisson_loss(self, target, pred):
        return (pred - target * self._log(pred)).mean()  

    def forward(self, pred, target):
        """
        Calculate the poisson loss for a given prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor.
        target : torch.Tensor
            Target tensor.

        Returns
        -------
        loss : torch.Tensor
            Poisson loss.
        """
        return self._poisson_loss(target, pred)
    
class BCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss for classification tasks. Wrapper around `torch.nn.BCEWithLogitsLoss`
    that takes care of the dimensionality of the input and target tensors.
    """
    def __init__(self):
        """
        Get a BCEWithLogitsLoss object that can be used to train a model.
        """
        super(BCEWithLogitsLoss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    
    def forward(self, pred, target, padding_value = -100):
        """
        Calculate the BCEWithLogitsLoss for a given prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor of logits.
        target : torch.Tensor
            Target tensor of labels.
        padding_value : int, optional
            Value to ignore in the loss calculation. The default is -100.

        Returns
        -------
        loss : torch.Tensor
            BCEWithLogitsLoss.
        """
        if pred.dim() == 3:
            loss =  self.criterion(pred.permute(0, 2, 1), target.float())
        else: 
            loss = self.criterion(pred, target.float())
        # remove loss for padded positions and return
        return torch.mean(loss[~target != padding_value])
    
class MSELoss(nn.Module):
    """
    MSE loss for regression tasks. Wrapper around `torch.nn.MSELoss`
    that takes care of the dimensionality of the input and target tensors.
    """
    def __init__(self):
        """
        Get a MSELoss object that can be used to train a model.
        """
        super(MSELoss, self).__init__()
    
    def forward(self, pred, target):
        """
        Calculate the MSE loss for a given prediction and target.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor.
        target : torch.Tensor
            Target tensor.

        Returns
        -------
        loss : torch.Tensor
            MSE loss.
        """
        criterion = torch.nn.MSELoss()
        return criterion(pred.permute(0, 2, 1), target)

class BaseTrainer:
    ''''Performs training and validation steps for a given model and dataset.
    We use hydra to configure the trainer. The configuration is passed to the
    trainer as an OmegaConf object.
    '''
    def __init__(self, 
                model, 
                optimizer, 
                criterion, 
                device, 
                config, 
                overwrite_dir=False, 
                gradient_accumulation_steps: int = 1, ):
        """
        Get a BaseTrainer object that can be used to train a model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        criterion : torch.nn.Module
            Loss function to use for training.
        device : torch.device
            Device to use for training.
        config : OmegaConf
            Configuration object.
        overwrite_dir : bool, optional
            Whether to overwrite the output directory. The default is False.
        gradient_accumulation_steps : int, optional
            Number of gradient accumulation steps. The default is 1.
        """


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
            pd.DataFrame(columns = ['Epoch', 'train_loss', 'val_loss', f'val_{self.config.params.metric}']).to_csv(f'{path}/losses.csv', index = False)
        return 
    
    def _load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint,  map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        val_metric = checkpoint[f'val_{self.config.params.metric}']
        return epoch, train_loss, val_loss, val_metric
    
    def _save_checkpoint(self, epoch, train_loss, val_loss, val_metric):
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
    
    def _log_wandb(self, epoch, train_loss, val_loss, val_metric):
        wandb.log({'train_loss': train_loss, 
                   'val_loss': val_loss, 
                   f'val_{self.config.params.metric}': val_metric}, 
                   step = epoch)
        
        # wandb.log({"Training latent with labels": wandb.Image(plt)})
        return
    
    def _calculate_metric(self, y_true, y_pred):
        ''' 
        Calculates the metric for the given task
        The metric calculated is specified in the config.params.metric
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            metric: the metric value
        '''
        # check if any padding in the target
        if torch.any(y_true  == self.config.data.padding_value):
            mask = y_true != self.config.data.padding_value
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        if self.config.params.metric == 'mcc':
            metric =  matthews_corrcoef(y_true.numpy().ravel(), y_pred.numpy().ravel())
    
        elif self.config.params.metric == 'auroc':
            if self.config.task == 'histone_modification' or self.config.task == 'chromatin_accessibility':
                # save y_true and y_pred 
                metric = roc_auc_score(y_true.numpy(), y_pred.numpy(), average = None)
            else:
                metric = roc_auc_score(y_true.numpy().ravel(), y_pred.numpy().ravel(), average = 'macro') # flatten arrays to get pearsons r
            
        elif self.config.params.metric == 'pearsonr':
            metric = r_regression(y_true.detach().numpy().reshape(-1,1), 
                                    y_pred.detach().numpy().ravel())[0] # flatten arrays to get pearsons r

        elif self.config.params.metric == 'auprc' :
            metric = average_precision_score(y_true.numpy().ravel(), y_pred.numpy().ravel(), average='macro')
            
        return metric
    

    def _get_checkpoint_path(self, 
                             load_checkpoint : Union[bool, int] = True):
        '''
        Gets the path of the checkpoint to load
        Args:
            load_checkpoint: if true, load latest checkpoint and continue training, if int, 
                            load checkpoint from that epoch and continue training
        Returns:
            checkpoint_path: the path of the checkpoint to load
        '''

        checkpoints = [f for f in os.listdir(f'{self.config.output_dir}/checkpoints/') if f.endswith('.pt')]
        if len(checkpoints) == 0 or not load_checkpoint:
            print('No checkpoints found, starting from scratch')
            return 
        else:
            if isinstance(load_checkpoint, bool):
                    print('Load latest checkpoint')
                    load_checkpoint = checkpoints[-1]
            elif isinstance(load_checkpoint, int):
                load_checkpoint = f'epoch_{load_checkpoint}.pt'
        
        checkpoint_path = f'{self.config.output_dir}/checkpoints/{load_checkpoint}'
        # check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'Checkpoint {checkpoint_path} does not exist')
        return checkpoint_path
        

    def train_epoch(self, train_loader): # one epoch
        """
        Performs one epoch of training.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data loader.

        Returns
        -------
        train_loss : float
            The average training loss for the epoch.
        """
        self.model.train()
        train_loss = 0
        for idx, batch in enumerate(train_loader):
            print('batch', idx)
            train_loss += self.train_step(batch, idx = idx)
        train_loss /= (idx +1)
        return train_loss
    

    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs, 
              load_checkpoint: Union[bool, int] = True):
        """
        Performs the full training routine.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data loader.
        val_loader : torch.utils.data.DataLoader
            The validation data loader.
            epochs : int
            The number of epochs to train for.
        load_checkpoint : bool, optional
            If True, loads the latest checkpoint from the output directory and
            continues training. If an integer is provided, loads the checkpoint
            from that epoch and continues training.
            
        Returns
        -------
        None
        """
        print('Training')
        # if load checkpoint is true, then load latest model and continue training
        start_epoch = 0
        checkpoint_path = self._get_checkpoint_path(load_checkpoint)
        if checkpoint_path:
            start_epoch, train_loss, val_loss, val_metric = self._load_checkpoint(checkpoint_path)
            print(f'Loaded checkpoint from epoch {start_epoch}, train loss: {train_loss}, val loss: {val_loss}, val {self.config.params.metric}: {val_metric}')

        for epoch in range(1+ start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metric = self.validate(val_loader)
            val_metric = np.mean(val_metric)
            # save epoch in output dir
            self._save_checkpoint(epoch, train_loss, val_loss, val_metric)
            # log losses to csv
            self._log_loss(epoch, train_loss, val_loss, val_metric)
            # log to wandb 
            self._log_wandb(epoch, train_loss, val_loss, val_metric)
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val {self.config.params.metric}: {val_metric:.4f}')
        return
    
    def train_step(self, batch, idx = 0):
        """
        Performs a single training step.
        
        Parameters
        ----------
        batch : tuple
            A tuple containing the batch of data and labels, as returned by the
            data loader.
        idx : int
            The index of the batch.
            
        Returns
        -------
        loss : float
            The loss for the batch.
        ß"""
        self.model.train()
        data, target = batch
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(data.to(self.device), length = target.shape[-1], 
                                activation = self.config.params.activation) 
            if self.config.task == 'chromatin_accessibility' or self.config.task == 'histone_modification':
                output = output.squeeze(1)
            loss = self.criterion(output, target.to(self.device).long())
            loss = loss / self.gradient_accumulation_steps
        # Accumulates scaled gradients.
        self.scaler.scale(loss).backward()
        if ((idx + 1) % self.gradient_accumulation_steps == 0) : #or (idx + 1 == len_dataloader):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none = True)
            
        return loss.item()

    def validate(self, data_loader, save_output = False):
        """
        Performs validation.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader to be used.
        save_output : bool, optional
            If True, saves the targets and outputs to a torch file.

        Returns
        -------
        loss : float
            The average validation loss.
        metric : float
            The average validation metric.
        """
        self.model.eval()
        loss = 0
        outputs = []
        targets_all = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):
                output = self.model(data.to(self.device), activation = self.config.params.activation)
                if self.config.task == 'chromatin_accessibility' or self.config.task == 'histone_modification':
                    output = output.squeeze(1)
                    outputs.append(self.model.sigmoid(output).detach().cpu())
                else: 
                    outputs.append(torch.argmax(self.model.softmax(output), dim=-1).detach().cpu()) 
                loss += self.criterion(output, target.to(self.device).long()).item()
                targets_all.append(target.detach().cpu())  

        loss /= (idx + 1) 
        if save_output:
            torch.save({'targets': targets_all, 'outputs': outputs}, f'{self.config.output_dir}/test_set.torch')
        # compute metrics
        metric = self._calculate_metric(torch.cat(targets_all), 
                                          torch.cat(outputs))
        
        return loss, metric

    def test(self, test_loader, checkpoint = None, overwrite=False):
        """
        Performs testing.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            The data loader to be used.
        checkpoint : pandas.DataFrame, optional
            The checkpoint to be used. If None, loads the checkpoint with the
            lowest validation loss.
        overwrite : bool, optional
            If True, overwrites the `best_model_metrics` file.

        Returns
        -------
        loss : float
            The average validation loss.
        metric : float
            The average validation metric.
        """
        df = pd.read_csv(f'{self.config.output_dir}/losses.csv')
        if checkpoint is None:
            checkpoint = pd.DataFrame(df.iloc[df["val_loss"].idxmin()]).T.reset_index(drop=True) 
        # load checkpoint
        print(f'{self.config.output_dir}/checkpoints/epoch_{int(checkpoint["Epoch"].iloc[0])}.pt')
        epoch, train_loss, val_loss, val_metric = self._load_checkpoint(f'{self.config.output_dir}/checkpoints/epoch_{int(checkpoint["Epoch"].iloc[0])}.pt')
        print(f'Loaded checkpoint from epoch {epoch}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, Val {self.config.params.metric}: {np.mean(val_metric):.3f}')

        # test
        loss, metric = self.validate(test_loader, save_output = False)
        
        print(f'Test results : Loss {loss:.4f}, {self.config.params.metric} {np.mean(metric):.4f}')
        
        if isinstance(metric, np.ndarray):
            columns = ['test_loss', f'test_{self.config.params.metric}_avg'] +[f'test_{self.config.params.metric}_{n}' for n in range(len(metric))]
            data = [[loss, np.mean(metric)] + list(metric)]
        else:
            columns = ['test_loss', f'test_{self.config.params.metric}']
            data = [[loss, metric]]
            
        metrics = checkpoint.merge(pd.DataFrame(data = data, columns = columns), how = 'cross')

        if not overwrite and os.path.exists(f'{self.config.output_dir}/best_model_metrics.csv'):
            best_model_metrics = pd.read_csv(f'{self.config.output_dir}/best_model_metrics.csv', index_col = False) 
            # concat metrics to best model metrics
            metrics = pd.concat([best_model_metrics, metrics], ignore_index=True)

        # save metrics to best model metrics
        metrics = metrics.drop_duplicates().reset_index(drop=True)
        metrics.to_csv(f'{self.config.output_dir}/best_model_metrics.csv', index = False)
        return loss, metric 
    

    