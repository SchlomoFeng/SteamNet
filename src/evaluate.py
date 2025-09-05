"""
Stage 3: Offline Evaluation
构建SteamNet项目的离线模型评估与报告生成脚本
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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


def load_artifacts(model_path: str, data_dir: str, device: str = 'cpu') -> Tuple[SteamNet_Autoencoder, Dict]:
    """
    加载工件：模型权重和必要的数据文件
    
    Args:
        model_path: 模型文件路径
        data_dir: 数据目录路径
        device: 计算设备
        
    Returns:
        (model, metadata)
    """
    print(f"Loading artifacts from {model_path} and {data_dir}")
    
    data_dir = Path(data_dir)
    
    # 加载模型配置
    config_path = Path(model_path).parent / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        feature_dims = config_data['feature_dims']
        model_config = config_data['config']
    else:
        # 如果没有配置文件，使用默认值
        print("Warning: No training config found, using default values")
        feature_dims = {
            'node_feature_dim': 12,
            'edge_feature_dim': 3,
            'window_size': 24,
            'num_nodes': 35,
            'num_edges': 25
        }
        model_config = {
            'hidden_dim': 64,
            'gru_hidden_dim': 32,
            'num_hgt_layers': 3,
            'num_gru_layers': 2,
            'dropout': 0.1
        }
    
    # 创建模型
    model = create_model(
        feature_dims,
        hidden_dim=model_config['hidden_dim'],
        gru_hidden_dim=model_config['gru_hidden_dim'],
        num_hgt_layers=model_config['num_hgt_layers'],
        num_gru_layers=model_config['num_gru_layers'],
        dropout=model_config['dropout']
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # 加载元数据
    metadata = {}
    
    # 加载节点和边映射
    with open(data_dir / "node_map.json", 'r', encoding='utf-8') as f:
        metadata['node_map'] = json.load(f)
    
    with open(data_dir / "link_map.json", 'r', encoding='utf-8') as f:
        metadata['link_map'] = json.load(f)
    
    # 加载邻接矩阵
    with open(data_dir / "adjacency.pkl", 'rb') as f:
        metadata['adjacency'] = pickle.load(f)
    
    print(f"✓ Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Loaded metadata for {len(metadata['node_map'])} nodes")
    
    return model, metadata


def run_inference(model: SteamNet_Autoencoder, test_loader, device: str = 'cpu') -> pd.DataFrame:
    """
    运行模型推理并计算重构误差
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        
    Returns:
        包含重构误差和标签的DataFrame
    """
    print("Running inference on test data...")
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Inference")):
            data = data.to(device)
            
            # 前向传播
            reconstructions = model(data)
            
            # 获取原始数据
            original_features = data['sensor_node'].x  # [batch, nodes, time, features]
            original_labels = data['sensor_node'].y    # [batch, nodes, time]
            
            batch_size, num_nodes, window_size, _ = original_features.shape
            
            # 计算每个特征的重构误差
            for b in range(batch_size):
                for n in range(num_nodes):
                    for t in range(window_size):
                        # 计算各个特征的重构误差
                        pressure_error = 0.0
                        flow_error = 0.0
                        temp_error = 0.0
                        
                        if 'sensor_node_pressure' in reconstructions:
                            pressure_pred = reconstructions['sensor_node_pressure'][b, n, t, 0]
                            pressure_true = original_features[b, n, t, 0]
                            pressure_error = float((pressure_pred - pressure_true) ** 2)
                        
                        if 'sensor_node_mass_flow' in reconstructions:
                            flow_pred = reconstructions['sensor_node_mass_flow'][b, n, t, 0]
                            flow_true = original_features[b, n, t, 1]
                            flow_error = float((flow_pred - flow_true) ** 2)
                        
                        if 'sensor_node_temperature' in reconstructions:
                            temp_pred = reconstructions['sensor_node_temperature'][b, n, t, 0]
                            temp_true = original_features[b, n, t, 2]
                            temp_error = float((temp_pred - temp_true) ** 2)
                        
                        # 总重构误差（各特征的加权平均）
                        total_error = pressure_error + flow_error + temp_error
                        
                        # 找出最大误差的特征
                        errors = {
                            'Pressure': pressure_error,
                            'Mass_Flow': flow_error,
                            'Temperature': temp_error
                        }
                        max_error_feature = max(errors, key=errors.get)
                        max_error_value = errors[max_error_feature]
                        
                        results.append({
                            'batch_idx': batch_idx,
                            'node_idx': n,
                            'time_idx': t,
                            'reconstruction_error': total_error,
                            'pressure_error': pressure_error,
                            'flow_error': flow_error,
                            'temp_error': temp_error,
                            'max_error_feature': max_error_feature,
                            'max_error_value': max_error_value,
                            'true_label': int(original_labels[b, n, t])
                        })
    
    results_df = pd.DataFrame(results)
    print(f"✓ Computed reconstruction errors for {len(results_df)} data points")
    
    return results_df


def determine_threshold(val_loader, model, device: str = 'cpu', percentile: float = 99.5) -> float:
    """
    使用验证集确定异常检测阈值
    
    Args:
        val_loader: 验证数据加载器
        model: 训练好的模型
        device: 计算设备
        percentile: 阈值百分位数
        
    Returns:
        异常检测阈值
    """
    print(f"Determining threshold using {percentile}th percentile on validation set...")
    
    val_results = run_inference(model, val_loader, device)
    
    # 只使用正常数据（label=0）计算阈值
    normal_errors = val_results[val_results['true_label'] == 0]['reconstruction_error']
    
    if len(normal_errors) == 0:
        print("Warning: No normal data found in validation set, using all data")
        normal_errors = val_results['reconstruction_error']
    
    threshold = np.percentile(normal_errors, percentile)
    
    print(f"✓ Threshold determined: {threshold:.6f}")
    print(f"  Based on {len(normal_errors)} normal samples")
    
    return threshold


def generate_report(results_df: pd.DataFrame, threshold: float, output_dir: str) -> Dict:
    """
    生成性能评估报告
    
    Args:
        results_df: 推理结果DataFrame
        threshold: 异常检测阈值
        output_dir: 输出目录
        
    Returns:
        评估指标字典
    """
    print(f"Generating evaluation report...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 计算预测标签
    results_df['predicted_label'] = (results_df['reconstruction_error'] > threshold).astype(int)
    
    # 过滤掉label=2的数据点（无需求状态），只评估normal(0)和anomaly(1)
    eval_mask = results_df['true_label'].isin([0, 1])
    eval_results = results_df[eval_mask].copy()
    
    if len(eval_results) == 0:
        print("Warning: No evaluable data points found")
        return {}
    
    # 计算分类指标
    y_true = eval_results['true_label']
    y_pred = eval_results['predicted_label']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(eval_results),
        'anomaly_samples': (y_true == 1).sum(),
        'detected_anomalies': (y_pred == 1).sum()
    }
    
    # 打印结果
    print("\n=== Evaluation Results ===")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Total samples: {len(eval_results)}")
    print(f"True anomalies: {(y_true == 1).sum()}")
    print(f"Detected anomalies: {(y_pred == 1).sum()}")
    
    # 生成异常事件摘要
    anomaly_events = []
    
    # 找出被检测为异常的事件
    detected_anomalies = eval_results[eval_results['predicted_label'] == 1]
    
    for _, row in detected_anomalies.iterrows():
        anomaly_events.append({
            'start_time': f"step_{row['time_idx']}",  # 简化的时间表示
            'end_time': f"step_{row['time_idx']}",
            'element_id': f"node_{row['node_idx']}",
            'element_type': 'sensor_node',
            'max_error': row['max_error_value'],
            'feature_with_max_error': row['max_error_feature'],
            'reconstruction_error': row['reconstruction_error']
        })
    
    # 保存摘要CSV
    if anomaly_events:
        summary_df = pd.DataFrame(anomaly_events)
        summary_path = output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Saved evaluation summary to {summary_path}")
    
    # 保存详细的评估指标
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        # 将numpy arrays转换为lists以便JSON序列化
        json_metrics = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in metrics.items()}
        json.dump(json_metrics, f, indent=2)
    
    print(f"✓ Saved evaluation metrics to {metrics_path}")
    
    return metrics\n\n\ndef visualize_results(results_df: pd.DataFrame, \n                     threshold: float, \n                     metadata: Dict, \n                     output_dir: str,\n                     num_examples: int = 2) -> None:\n    \"\"\"\n    可视化异常检测结果\n    \n    Args:\n        results_df: 推理结果DataFrame\n        threshold: 异常检测阈值\n        metadata: 元数据字典\n        output_dir: 输出目录\n        num_examples: 生成示例数量\n    \"\"\"\n    print(f\"Creating visualization examples...\")\n    \n    output_dir = Path(output_dir)\n    figures_dir = output_dir / \"figures\"\n    figures_dir.mkdir(exist_ok=True)\n    \n    # 找出一些典型的被正确检测的异常事件\n    detected_anomalies = results_df[\n        (results_df['true_label'] == 1) & \n        (results_df['reconstruction_error'] > threshold)\n    ]\n    \n    if len(detected_anomalies) == 0:\n        print(\"Warning: No correctly detected anomalies found for visualization\")\n        # 改为显示一些有趣的正常数据\n        detected_anomalies = results_df[\n            results_df['reconstruction_error'] > threshold\n        ].head(num_examples)\n    \n    examples_to_plot = detected_anomalies.head(num_examples)\n    \n    for idx, (_, example) in enumerate(examples_to_plot.iterrows()):\n        node_idx = int(example['node_idx'])\n        time_idx = int(example['time_idx'])\n        \n        # 获取该节点在时间窗口内的所有数据\n        node_data = results_df[\n            (results_df['node_idx'] == node_idx) &\n            (results_df['batch_idx'] == example['batch_idx'])\n        ].sort_values('time_idx')\n        \n        if len(node_data) == 0:\n            continue\n        \n        # 创建图表\n        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))\n        \n        time_steps = node_data['time_idx']\n        \n        # 子图A: 异常分数随时间变化\n        ax1.plot(time_steps, node_data['reconstruction_error'], \n                 'b-', linewidth=2, label='Reconstruction Error')\n        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Threshold')\n        ax1.scatter([time_idx], [example['reconstruction_error']], \n                   color='red', s=100, zorder=5, label='Detected Anomaly')\n        \n        ax1.set_xlabel('Time Step')\n        ax1.set_ylabel('Reconstruction Error')\n        ax1.set_title('Anomaly Score Over Time')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        \n        # 子图B: 最大误差特征的实际值vs重构值\n        # 这里简化处理，显示该特征的误差\n        max_error_feature = example['max_error_feature']\n        \n        if max_error_feature == 'Pressure':\n            feature_errors = node_data['pressure_error']\n        elif max_error_feature == 'Mass_Flow':\n            feature_errors = node_data['flow_error']\n        elif max_error_feature == 'Temperature':\n            feature_errors = node_data['temp_error']\n        else:\n            feature_errors = node_data['reconstruction_error']\n        \n        ax2.plot(time_steps, feature_errors, \n                 'g-', linewidth=2, label=f'{max_error_feature} Error')\n        ax2.scatter([time_idx], [example['max_error_value']], \n                   color='red', s=100, zorder=5, label='Max Error Point')\n        \n        ax2.set_xlabel('Time Step')\n        ax2.set_ylabel('Feature Error')\n        ax2.set_title(f'{max_error_feature} Error Over Time')\n        ax2.legend()\n        ax2.grid(True, alpha=0.3)\n        \n        # 设置总标题\n        event_time = f\"Step {time_idx}\"\n        fig.suptitle(f'Anomaly Event on Node {node_idx} at {event_time}', \n                    fontsize=14, fontweight='bold')\n        \n        plt.tight_layout()\n        \n        # 保存图表\n        filename = f\"anomaly_event_node_{node_idx}_time_{time_idx}.png\"\n        filepath = figures_dir / filename\n        plt.savefig(filepath, dpi=150, bbox_inches='tight')\n        plt.close()\n        \n        print(f\"✓ Saved visualization: {filename}\")\n    \n    # 创建一个总览图：错误分布直方图\n    plt.figure(figsize=(10, 6))\n    \n    # 正常数据的错误分布\n    normal_errors = results_df[results_df['true_label'] == 0]['reconstruction_error']\n    anomaly_errors = results_df[results_df['true_label'] == 1]['reconstruction_error']\n    \n    plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)\n    if len(anomaly_errors) > 0:\n        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)\n    \n    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label='Threshold')\n    \n    plt.xlabel('Reconstruction Error')\n    plt.ylabel('Density')\n    plt.title('Distribution of Reconstruction Errors')\n    plt.legend()\n    plt.grid(True, alpha=0.3)\n    \n    overview_path = figures_dir / \"error_distribution.png\"\n    plt.savefig(overview_path, dpi=150, bbox_inches='tight')\n    plt.close()\n    \n    print(f\"✓ Saved error distribution plot: {overview_path}\")\n\n\ndef main():\n    \"\"\"主函数：执行完整的离线评估流程\"\"\"\n    print(\"Starting Stage 3: Offline Evaluation\")\n    \n    # 设置路径\n    base_dir = Path(__file__).parent.parent\n    data_dir = base_dir / \"data\" / \"processed\"\n    models_dir = base_dir / \"models\"\n    reports_dir = base_dir / \"reports\"\n    \n    model_path = models_dir / \"best_steamnet_model.pt\"\n    \n    # 检查必要文件\n    if not model_path.exists():\n        print(f\"Error: Model file not found at {model_path}\")\n        print(\"Please run Stage 2 (train.py) first to train the model.\")\n        return\n    \n    if not data_dir.exists():\n        print(f\"Error: Processed data not found at {data_dir}\")\n        print(\"Please run Stage 1 (build_features.py) first to process the data.\")\n        return\n    \n    # 设置设备\n    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n    print(f\"Using device: {device}\")\n    \n    try:\n        # 1. 加载工件\n        model, metadata = load_artifacts(str(model_path), str(data_dir), device)\n        \n        # 2. 准备测试数据\n        print(\"\\nPreparing test data...\")\n        train_loader, val_loader, test_loader = create_data_loaders(\n            str(data_dir), \n            window_size=24,\n            batch_size=16\n        )\n        \n        # 3. 确定阈值\n        print(\"\\nDetermining threshold...\")\n        threshold = determine_threshold(val_loader, model, device)\n        \n        # 4. 运行推理\n        print(\"\\nRunning inference on test set...\")\n        results_df = run_inference(model, test_loader, device)\n        \n        # 5. 生成评估报告\n        print(\"\\nGenerating evaluation report...\")\n        metrics = generate_report(results_df, threshold, str(reports_dir))\n        \n        # 6. 创建可视化\n        print(\"\\nCreating visualizations...\")\n        visualize_results(results_df, threshold, metadata, str(reports_dir))\n        \n        print(\"\\n\" + \"=\"*50)\n        print(\"Stage 3 completed successfully!\")\n        print(f\"Reports saved to: {reports_dir}\")\n        print(\"\\nKey Results:\")\n        if metrics:\n            print(f\"  Precision: {metrics.get('precision', 0):.4f}\")\n            print(f\"  Recall: {metrics.get('recall', 0):.4f}\")\n            print(f\"  F1-Score: {metrics.get('f1_score', 0):.4f}\")\n            print(f\"  Threshold: {metrics.get('threshold', 0):.6f}\")\n        print(\"=\"*50)\n        \n    except Exception as e:\n        print(f\"Error in Stage 3: {e}\")\n        raise\n\n\nif __name__ == \"__main__\":\n    main()