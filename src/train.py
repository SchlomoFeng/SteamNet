"""
Stage 2: Model Development & Training - Training Script
构建SteamNet项目的模型训练、验证与保存脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

try:
    from .dataset import create_data_loaders, SteamNetDataset
    from .model import create_model, SteamNet_Autoencoder
except ImportError:
    from dataset import create_data_loaders, SteamNetDataset
    from model import create_model, SteamNet_Autoencoder


class MaskedMSELoss(nn.Module):
    """带掩码的均方误差损失"""
    
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算带掩码的MSE损失
        
        Args:
            pred: 预测值
            target: 目标值
            mask: 掩码，True表示有效位置
            
        Returns:
            损失值
        """
        if mask is None:
            return nn.functional.mse_loss(pred, target)
        
        # 只计算有效位置的损失
        valid_mask = mask & ~torch.isnan(target) & ~torch.isnan(pred)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        diff = (pred - target) ** 2
        masked_diff = diff[valid_mask]
        return masked_diff.mean()


class Trainer:
    """SteamNet模型训练器"""
    
    def __init__(self, 
                 model: SteamNet_Autoencoder,
                 train_loader,
                 val_loader,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化训练器
        
        Args:
            model: SteamNet模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.7, verbose=True
        )
        
        # 损失函数
        self.masked_mse = MaskedMSELoss()
        
        # 训练历史
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, data in enumerate(progress_bar):
            # 移动数据到设备
            data = data.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            reconstructions = self.model(data)
            
            # 计算损失
            loss, losses = self.model.compute_loss(reconstructions, data)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {'total_loss': avg_loss, **avg_components}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for batch_idx, data in enumerate(progress_bar):
                # 移动数据到设备
                data = data.to(self.device)
                
                # 前向传播
                reconstructions = self.model(data)
                
                # 计算损失
                loss, losses = self.model.compute_loss(reconstructions, data)
                
                # 累计损失
                total_loss += loss.item()
                for key, value in losses.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
                
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {'total_loss': avg_loss, **avg_components}
    
    def train(self, num_epochs: int, save_dir: str = "models") -> Dict:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            
        Returns:
            训练历史
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate_epoch()
            
            # 记录历史
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # 学习率调度
            self.scheduler.step(val_metrics['total_loss'])
            
            # 保存最佳模型
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_epoch = epoch
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }, save_path / "best_steamnet_model.pt")
                
                print(f"✓ New best model saved (val_loss: {self.best_val_loss:.4f})")
            
            # 打印训练信息
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} ({epoch_time:.1f}s) "
                  f"| Train: {train_metrics['total_loss']:.4f} "
                  f"| Val: {val_metrics['total_loss']:.4f} "
                  f"| LR: {current_lr:.2e}")
            
            # 详细损失信息
            if (epoch + 1) % 5 == 0:
                print("  Train components:", {k: f"{v:.4f}" for k, v in train_metrics.items() if k != 'total_loss'})
                print("  Val components:  ", {k: f"{v:.4f}" for k, v in val_metrics.items() if k != 'total_loss'})
            
            # 早停检查
            if epoch - self.best_epoch > 15:  # 15个epoch没有改善就早停
                print(f"Early stopping at epoch {epoch+1} (best was epoch {self.best_epoch+1})")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }


def main():
    """主训练函数"""
    print("Starting Stage 2: Model Development & Training")
    
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # 训练超参数
    config = {
        'window_size': 24,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'hidden_dim': 64,
        'gru_hidden_dim': 32,
        'num_hgt_layers': 3,
        'num_gru_layers': 2,
        'dropout': 0.1
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # 创建数据加载器
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            str(data_dir), 
            window_size=config['window_size'],
            batch_size=config['batch_size']
        )
        
        # 获取特征维度
        temp_dataset = SteamNetDataset(str(data_dir), window_size=config['window_size'], split='train')
        feature_dims = temp_dataset.get_feature_dimensions()
        
        print("\nFeature dimensions:")
        for key, value in feature_dims.items():
            print(f"  {key}: {value}")
        
        # 创建模型
        print("\nCreating model...")
        model = create_model(
            feature_dims,
            hidden_dim=config['hidden_dim'],
            gru_hidden_dim=config['gru_hidden_dim'],
            num_hgt_layers=config['num_hgt_layers'],
            num_gru_layers=config['num_gru_layers'],
            dropout=config['dropout']
        )
        
        # 创建训练器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            device=device
        )
        
        # 开始训练
        print("\nStarting training...")
        history = trainer.train(
            num_epochs=config['num_epochs'],
            save_dir=str(models_dir)
        )
        
        # 保存训练配置和历史
        config_path = models_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'config': config,
                'feature_dims': feature_dims,
                'history': history
            }, f, indent=2)
        
        print(f"\nTraining configuration and history saved to: {config_path}")
        print(f"Best model saved to: {models_dir / 'best_steamnet_model.pt'}")
        
        # 在测试集上评估最终模型
        print("\nEvaluating on test set...")
        model.eval()
        test_loss = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                reconstructions = model(data)
                loss, _ = model.compute_loss(reconstructions, data)
                test_loss += loss.item()
                num_test_batches += 1
        
        avg_test_loss = test_loss / num_test_batches
        print(f"Test loss: {avg_test_loss:.4f}")
        
        print("\nStage 2 completed successfully!")
        
    except Exception as e:
        print(f"Error in Stage 2: {e}")
        raise


if __name__ == "__main__":
    main()