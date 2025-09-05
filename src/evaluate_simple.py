"""
Stage 3: Offline Evaluation - Simplified Version
构建SteamNet项目的离线模型评估与报告生成脚本
"""

import sys
from pathlib import Path
# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

try:
    from .dataset import SteamNetDataset, create_data_loaders
    from .model import SteamNet_Autoencoder, create_model
except ImportError:
    from dataset import SteamNetDataset, create_data_loaders
    from model import SteamNet_Autoencoder, create_model


def run_simple_evaluation():
    """简化版评估函数"""
    print("Starting Stage 3: Offline Evaluation (Simplified)")
    
    # 设置路径
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "best_steamnet_model.pt"
    
    # 检查必要文件
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # 1. 加载模型配置
        config_path = models_dir / "training_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            feature_dims = config_data['feature_dims']
            model_config = config_data['config']
        else:
            # 默认配置
            feature_dims = {
                'node_feature_dim': 12,
                'edge_feature_dim': 3,
                'window_size': 24,
                'num_nodes': 140,  # This is batch*35 from our data
                'num_edges': 25
            }
            model_config = {
                'hidden_dim': 32,
                'gru_hidden_dim': 16,
                'num_hgt_layers': 2,
                'num_gru_layers': 1,
                'dropout': 0.1
            }
        
        # 2. 创建和加载模型
        print("Loading model...")
        model = create_model(
            feature_dims,
            hidden_dim=model_config['hidden_dim'],
            gru_hidden_dim=model_config['gru_hidden_dim'],
            num_hgt_layers=model_config['num_hgt_layers'],
            num_gru_layers=model_config['num_gru_layers'],
            dropout=model_config['dropout']
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # 3. 准备数据
        print("Preparing data...")
        train_loader, val_loader, test_loader = create_data_loaders(
            str(data_dir), window_size=24, batch_size=8
        )
        
        # 4. 确定阈值（使用验证集）
        print("Determining threshold...")
        val_errors = []
        model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if i >= 5:  # 限制计算量
                    break
                data = data.to(device)
                reconstructions = model(data)
                loss, _ = model.compute_loss(reconstructions, data)
                val_errors.append(loss.item())
        
        threshold = np.percentile(val_errors, 95.0) if val_errors else 1000.0
        print(f"✓ Threshold: {threshold:.6f}")
        
        # 5. 运行测试评估
        print("Running test evaluation...")
        test_results = []
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                if i >= 10:  # 限制计算量
                    break
                    
                data = data.to(device)
                reconstructions = model(data)
                loss, losses = model.compute_loss(reconstructions, data)
                
                # 简化：每个批次记录一个结果
                test_results.append({
                    'batch_idx': i,
                    'reconstruction_error': loss.item(),
                    'threshold': threshold,
                    'is_anomaly': loss.item() > threshold,
                    'pressure_loss': losses.get('pressure', 0),
                    'flow_loss': losses.get('mass_flow', 0),
                    'temp_loss': losses.get('temperature', 0)
                })
        
        # 6. 生成报告
        results_df = pd.DataFrame(test_results)
        
        # 计算检测统计
        total_samples = len(results_df)
        detected_anomalies = results_df['is_anomaly'].sum()
        detection_rate = detected_anomalies / total_samples if total_samples > 0 else 0
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total test batches: {total_samples}")
        print(f"Detected anomalies: {detected_anomalies}")
        print(f"Detection rate: {detection_rate:.2%}")
        print(f"Threshold used: {threshold:.6f}")
        print(f"Average reconstruction error: {results_df['reconstruction_error'].mean():.6f}")
        print(f"Max reconstruction error: {results_df['reconstruction_error'].max():.6f}")
        print(f"Min reconstruction error: {results_df['reconstruction_error'].min():.6f}")
        
        # 7. 保存结果
        results_df.to_csv(reports_dir / "evaluation_summary.csv", index=False)
        
        # 保存评估指标
        metrics = {
            'threshold': float(threshold),
            'total_samples': int(total_samples),
            'detected_anomalies': int(detected_anomalies),
            'detection_rate': float(detection_rate),
            'avg_reconstruction_error': float(results_df['reconstruction_error'].mean()),
            'max_reconstruction_error': float(results_df['reconstruction_error'].max()),
            'min_reconstruction_error': float(results_df['reconstruction_error'].min())
        }
        
        with open(reports_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 8. 创建简单可视化
        print("Creating visualizations...")
        figures_dir = reports_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 重构误差分布图
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['reconstruction_error'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(figures_dir / "error_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 异常检测结果图
        plt.figure(figsize=(12, 6))
        batch_indices = results_df['batch_idx']
        errors = results_df['reconstruction_error']
        colors = ['red' if x else 'blue' for x in results_df['is_anomaly']]
        
        plt.scatter(batch_indices, errors, c=colors, alpha=0.7, s=50)
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        plt.xlabel('Batch Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Anomaly Detection Results')
        plt.legend(['Normal', 'Anomaly', 'Threshold'])
        plt.grid(True, alpha=0.3)
        plt.savefig(figures_dir / "anomaly_detection_results.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Results saved to: {reports_dir}")
        print(f"✓ Figures saved to: {figures_dir}")
        print("="*50)
        print("Stage 3 completed successfully!")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_simple_evaluation()