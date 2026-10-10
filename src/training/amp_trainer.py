import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
import os


class TEMDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.from_numpy(data).float()
        if len(self.data.shape) == 2:
            self.data = self.data.unsqueeze(1)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class AMPScaler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: optim.Optimizer):
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def update(self):
        if self.enabled and self.scaler is not None:
            self.scaler.update()


class AMPEngine:
    def __init__(self, model: nn.Module, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model = model.to(self.device)
        self.scaler = AMPScaler()

    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                    loss_fn: nn.Module) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for data, labels in tqdm(dataloader, desc='Training'):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.scaler.enabled):
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = loss_fn(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            
            total_loss += loss.item() * data.size(0)
            n_samples += data.size(0)
        
        return total_loss / n_samples

    def validate_epoch(self, dataloader: DataLoader, loss_fn: nn.Module) -> float:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for data, labels in tqdm(dataloader, desc='Validation'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                with autocast(enabled=self.scaler.enabled):
                    outputs = self.model(data)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = loss_fn(outputs, labels)
                
                total_loss += loss.item() * data.size(0)
                n_samples += data.size(0)
        
        return total_loss / n_samples

    def test(self, dataloader: DataLoader, loss_fn: nn.Module) -> Dict:
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, labels in tqdm(dataloader, desc='Testing'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                with autocast(enabled=self.scaler.enabled):
                    outputs = self.model(data)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = loss_fn(outputs, labels)
                
                total_loss += loss.item() * data.size(0)
                predictions.append(outputs.cpu().numpy())
                targets.append(labels.cpu().numpy())
        
        return {
            'loss': total_loss / len(dataloader.dataset),
            'predictions': np.concatenate(predictions),
            'targets': np.concatenate(targets)
        }


class Trainer:
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.optimizer = self._create_optimizer()
        self.loss_fn = self._create_loss_fn()
        self.scheduler = self._create_scheduler()
        self.amp_engine = AMPEngine(model, self.device)
        
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def _create_optimizer(self) -> optim.Optimizer:
        opt_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        
        if opt_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr)
        elif opt_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            return optim.Adam(self.model.parameters(), lr=lr)

    def _create_loss_fn(self) -> nn.Module:
        loss_type = self.config.get('loss', 'mse')
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        else:
            return nn.MSELoss()

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        scheduler_type = self.config.get('scheduler', None)
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        else:
            return None

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, epoch_callback=None,
              checkpoint_dir: str = '.') -> Dict:
        """训练主循环。

        Args:
            epoch_callback: 可选回调 fn(epoch, train_loss, val_loss)，
                            返回 False 时提前终止（供 GUI 训练监控/停止按钮使用）。
            checkpoint_dir: 最优模型 checkpoint 保存目录。
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = self.amp_engine.train_epoch(train_loader, self.optimizer, self.loss_fn)
            val_loss = self.amp_engine.validate_epoch(val_loader, self.loss_fn)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint(os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth'))
            else:
                self.early_stopping_counter += 1

            if epoch_callback is not None and epoch_callback(epoch + 1, train_loss, val_loss) is False:
                print("Training stopped by callback")
                break

            if self.config.get('early_stopping', False) and self.early_stopping_counter >= 10:
                print("Early stopping triggered")
                break

        return history

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']