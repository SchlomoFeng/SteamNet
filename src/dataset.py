"""
Stage 2: Model Development & Training - Dataset Module
构建SteamNet项目的PyG Dataset和DataLoader定义
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Dataset, HeteroData
from sklearn.preprocessing import StandardScaler
import joblib


class SteamNetDataset(Dataset):
    """
    SteamNet数据集类，继承自torch_geometric.data.Dataset
    用于加载和处理蒸汽管网时序数据
    """
    
    def __init__(self, data_dir: str, window_size: int = 24, split: str = 'train', transform=None, pre_transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 处理后的数据目录路径
            window_size: 时间窗口大小
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据变换函数
            pre_transform: 预处理变换函数
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.split = split
        
        # 加载所有必要文件
        self._load_metadata()
        self._load_features()
        self._prepare_data_splits()
        
        super(SteamNetDataset, self).__init__(None, transform, pre_transform)
    
    def _load_metadata(self):
        """加载元数据文件"""
        print(f"Loading metadata from {self.data_dir}")
        
        # 加载节点和边的映射
        with open(self.data_dir / "node_map.json", 'r', encoding='utf-8') as f:
            self.node_map = json.load(f)
        
        with open(self.data_dir / "link_map.json", 'r', encoding='utf-8') as f:
            self.link_map = json.load(f)
        
        # 加载邻接矩阵
        with open(self.data_dir / "adjacency.pkl", 'rb') as f:
            self.edge_index_dict = pickle.load(f)
        
        # 加载静态特征缩放器
        self.static_scaler = joblib.load(self.data_dir / "static_feature_scaler.joblib")
        
        print(f"Loaded {len(self.node_map)} nodes and {len(self.link_map)} links")
    
    def _load_features(self):
        """加载特征数据"""
        print("Loading feature data...")
        
        # 加载静态边特征
        self.static_edge_features = pd.read_parquet(self.data_dir / "static_edge_features.parquet")
        
        # 加载时序节点特征
        self.node_features_df = pd.read_parquet(self.data_dir / "timeseries_node_features.parquet")
        
        # 加载时序边特征
        edge_features_file = self.data_dir / "timeseries_edge_features.parquet"
        if edge_features_file.exists():
            self.edge_features_df = pd.read_parquet(edge_features_file)
        else:
            self.edge_features_df = pd.DataFrame()
        
        # 对时序数据按时间排序
        self.node_features_df = self.node_features_df.sort_values('timestamp')
        
        print(f"Node features shape: {self.node_features_df.shape}")
        print(f"Edge features shape: {self.edge_features_df.shape}")
    
    def _prepare_data_splits(self):
        """准备数据集划分"""
        # 获取所有时间戳
        all_timestamps = sorted(self.node_features_df['timestamp'].unique())
        total_windows = len(all_timestamps) - self.window_size + 1
        
        if total_windows <= 0:
            raise ValueError(f"Not enough data for window_size={self.window_size}")
        
        # 数据划分：70% train, 15% val, 15% test
        train_end = int(0.7 * total_windows)
        val_end = int(0.85 * total_windows)
        
        self.timestamps = all_timestamps
        
        if self.split == 'train':
            self.valid_indices = list(range(0, train_end))
            # 对于训练，只使用label为0或2的数据（正常和无需求）
            self._filter_training_data()
        elif self.split == 'val':
            self.valid_indices = list(range(train_end, val_end))
        elif self.split == 'test':
            self.valid_indices = list(range(val_end, total_windows))
        else:
            raise ValueError(f"Invalid split: {self.split}")
        print(f"Dataset split '{self.split}': {len(self.valid_indices)} windows available")
    
    def _filter_training_data(self):
        """为训练集过滤异常数据"""
        if self.split != 'train':
            return
        
        # 只保留那些窗口内主要包含正常数据的时间窗口
        filtered_indices = []
        
        for idx in self.valid_indices:
            window_timestamps = self.timestamps[idx:idx + self.window_size]
            window_data = self.node_features_df[
                self.node_features_df['timestamp'].isin(window_timestamps)
            ]
            
            # 计算异常数据的比例
            anomaly_ratio = (window_data['label'] == 1).sum() / len(window_data)
            
            # 只保留异常比例低于阈值的窗口
            if anomaly_ratio < 0.1:  # 异常数据不超过10%
                filtered_indices.append(idx)
        
        self.valid_indices = filtered_indices
        print(f"After filtering: {len(self.valid_indices)} training windows remain")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> HeteroData:
        """
        获取单个数据样本
        
        Args:
            idx: 数据索引
        
        Returns:
            HeteroData对象
        """
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range")
        
        window_idx = self.valid_indices[idx]
        return self._create_hetero_data(window_idx)
    
    def _create_hetero_data(self, window_idx: int) -> HeteroData:
        """
        创建HeteroData对象
        
        Args:
            window_idx: 时间窗口索引
        
        Returns:
            HeteroData对象
        """
        # 获取时间窗口
        window_timestamps = self.timestamps[window_idx:window_idx + self.window_size]
        
        # 提取窗口内的节点特征
        window_node_data = self.node_features_df[
            self.node_features_df['timestamp'].isin(window_timestamps)
        ].copy()
        
        # 创建HeteroData对象
        data = HeteroData()
        
        # 准备节点特征
        self._add_node_features(data, window_node_data)
        
        # 准备边特征和连接关系
        self._add_edge_features(data, window_timestamps)
        
        return data
    
    def _add_node_features(self, data: HeteroData, window_node_data: pd.DataFrame):
        """添加节点特征到HeteroData"""
        # 选择特征列（排除元数据列）
        feature_columns = [col for col in window_node_data.columns 
                          if col not in ['Node_ID', 'timestamp', 'label']]
        
        # 为每个节点准备时序特征
        node_features_dict = {}
        node_labels_dict = {}
        
        for node_name in self.node_map.keys():
            node_data = window_node_data[window_node_data['Node_ID'] == node_name]
            
            if not node_data.empty:
                # 按时间排序
                node_data = node_data.sort_values('timestamp')
                
                # 提取特征（时间序列）
                features = node_data[feature_columns].values
                labels = node_data['label'].values
                
                # 如果数据不足window_size，进行填充
                if len(features) < self.window_size:
                    # 用最后一个值填充
                    padding_size = self.window_size - len(features)
                    if len(features) > 0:
                        last_feature = features[-1:].repeat(padding_size, axis=0)
                        features = np.vstack([features, last_feature])
                        last_label = labels[-1:].repeat(padding_size)
                        labels = np.concatenate([labels, last_label])
                    else:
                        # 如果完全没有数据，用零填充
                        features = np.zeros((self.window_size, len(feature_columns)))
                        labels = np.zeros(self.window_size)
            else:
                # 节点没有数据，用零填充
                features = np.zeros((self.window_size, len(feature_columns)))
                labels = np.zeros(self.window_size)
            
            node_features_dict[node_name] = features
            node_labels_dict[node_name] = labels
        
        # 添加到HeteroData（简化：所有节点都作为sensor_node类型）
        all_node_features = []
        all_node_labels = []
        
        for node_name in sorted(self.node_map.keys()):
            all_node_features.append(node_features_dict[node_name])
            all_node_labels.append(node_labels_dict[node_name])
        
        # 转换为张量
        node_features_tensor = torch.FloatTensor(np.stack(all_node_features))  # [num_nodes, window_size, num_features]
        node_labels_tensor = torch.LongTensor(np.stack(all_node_labels))      # [num_nodes, window_size]
        
        data['sensor_node'].x = node_features_tensor
        data['sensor_node'].y = node_labels_tensor
        
        # 添加节点数量信息
        data['sensor_node'].num_nodes = len(self.node_map)
    
    def _add_edge_features(self, data: HeteroData, window_timestamps: List):
        """添加边特征和连接关系到HeteroData"""
        # 添加边连接关系
        for edge_type, edge_index in self.edge_index_dict.items():
            edge_index_tensor = torch.LongTensor(edge_index)
            data[edge_type].edge_index = edge_index_tensor
            
            # 添加静态边特征（扩展到时间维度）
            if not self.static_edge_features.empty:
                num_edges = edge_index.shape[1]
                static_features = self.static_edge_features.values[:num_edges]  # 取前num_edges个特征
                
                # 扩展到window_size
                edge_features = np.tile(static_features[:, np.newaxis, :], 
                                      (1, self.window_size, 1))
                edge_features_tensor = torch.FloatTensor(edge_features)
                data[edge_type].edge_attr = edge_features_tensor
            
            # 如果有动态边特征，也添加进去
            if not self.edge_features_df.empty:
                # 这里可以添加动态边特征的处理逻辑
                pass
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """获取特征维度信息"""
        # 计算节点特征维度
        feature_columns = [col for col in self.node_features_df.columns 
                          if col not in ['Node_ID', 'timestamp', 'label']]
        node_feature_dim = len(feature_columns)
        
        # 计算边特征维度
        edge_feature_dim = self.static_edge_features.shape[1] if not self.static_edge_features.empty else 0
        
        return {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'window_size': self.window_size,
            'num_nodes': len(self.node_map),
            'num_edges': len(self.link_map)
        }


def create_data_loaders(data_dir: str, window_size: int = 24, batch_size: int = 32) -> Tuple:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据目录
        window_size: 时间窗口大小
        batch_size: 批次大小
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader
    
    # 创建数据集
    train_dataset = SteamNetDataset(data_dir, window_size, split='train')
    val_dataset = SteamNetDataset(data_dir, window_size, split='val')
    test_dataset = SteamNetDataset(data_dir, window_size, split='test')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    from pathlib import Path
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # 创建数据集实例
    dataset = SteamNetDataset(data_dir, window_size=24, split='train')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimensions: {dataset.get_feature_dimensions()}")
    
    # 测试获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Node features shape: {sample['sensor_node'].x.shape}")
        print(f"Node labels shape: {sample['sensor_node'].y.shape}")
        
        for edge_type in sample.edge_types:
            print(f"Edge type {edge_type}: {sample[edge_type].edge_index.shape}")