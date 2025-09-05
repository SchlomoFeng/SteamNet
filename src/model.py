"""
Stage 2: Model Development & Training - Model Architecture
构建SteamNet_Autoencoder模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional


class SteamNet_Autoencoder(nn.Module):
    """
    SteamNet自编码器模型
    使用异构图转换器(HGT)和GRU的时序图自编码器
    """
    
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int = 64,
                 gru_hidden_dim: int = 32,
                 num_hgt_layers: int = 3,
                 num_gru_layers: int = 2,
                 window_size: int = 24,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        初始化SteamNet自编码器
        
        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            hidden_dim: 隐藏层维度
            gru_hidden_dim: GRU隐藏层维度
            num_hgt_layers: HGT层数
            num_gru_layers: GRU层数
            window_size: 时间窗口大小
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super(SteamNet_Autoencoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.window_size = window_size
        self.dropout = dropout
        
        # 定义节点和边类型
        self.node_types = ['sensor_node']
        self.edge_types = [('sensor_node', 'connects_to', 'sensor_node')]
        
        # 输入投影层
        self.node_input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GCN编码器层（替代HGT）
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_hgt_layers):
            gcn_layer = GCNConv(hidden_dim, hidden_dim)
            self.gcn_layers.append(gcn_layer)
        
        # 为每种节点类型创建GRU层
        self.node_grus = nn.ModuleDict()
        for node_type in self.node_types:
            self.node_grus[node_type] = nn.GRU(
                input_size=hidden_dim,
                hidden_size=gru_hidden_dim,
                num_layers=num_gru_layers,
                batch_first=True,
                dropout=dropout if num_gru_layers > 1 else 0
            )
        
        # 解码器头部 - 为不同特征创建独立的MLP
        self.node_decoders = nn.ModuleDict()
        
        # 主要物理量解码器
        self.node_decoders['pressure'] = self._create_decoder_head(gru_hidden_dim, 1)
        self.node_decoders['mass_flow'] = self._create_decoder_head(gru_hidden_dim, 1)
        self.node_decoders['temperature'] = self._create_decoder_head(gru_hidden_dim, 1)
        
        # 导数特征解码器
        self.node_decoders['derivatives'] = self._create_decoder_head(gru_hidden_dim, 3)  # dP/dt, dT/dt, dM/dt
        
        # 掩码解码器
        self.node_decoders['masks'] = self._create_decoder_head(gru_hidden_dim, 3)  # pressure_mask, flow_mask, temp_mask
        
        # 状态解码器
        self.node_decoders['states'] = self._create_decoder_head(gru_hidden_dim, 3)  # state_normal, state_anomaly, state_no_demand
        
        # 边特征解码器（如果有边特征）
        if edge_feature_dim > 0:
            self.edge_decoders = nn.ModuleDict()
            self.edge_decoders['static_features'] = self._create_decoder_head(gru_hidden_dim, edge_feature_dim)
    
    def _create_decoder_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """创建解码器头部"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: HeteroData对象
            
        Returns:
            重构结果字典
        """
        # Get the correct tensor dimensions
        node_features = data['sensor_node'].x  # This should be [batch_size * num_nodes, window_size, feature_dim]
        if len(node_features.shape) == 3:
            # Batch format: [batch_size * num_nodes, window_size, feature_dim]
            batch_nodes, window_size, feature_dim = node_features.shape
            num_nodes = data['sensor_node'].num_nodes
            batch_size = batch_nodes // num_nodes
            # Reshape to [batch_size, num_nodes, window_size, feature_dim]
            node_features = node_features.view(batch_size, num_nodes, window_size, feature_dim)
        else:
            batch_size, num_nodes, window_size, feature_dim = node_features.shape
        
        # 存储所有时间步的编码结果
        temporal_encodings = []
        
        # 对每个时间步进行GCN编码
        for t in range(window_size):
            # 提取当前时间步的特征
            node_t_features = node_features[:, :, t, :]  # [batch_size, num_nodes, feature_dim]
            batch_size_t, num_nodes_t, feature_dim_t = node_t_features.shape
            
            # 重新整形和投影
            x = node_t_features.view(-1, feature_dim_t)  # [batch_size * num_nodes, feature_dim]
            x = self.node_input_proj(x)  # 投影到隐藏维度
            
            # 边连接信息（为每个批次复制）
            edge_index = data[('sensor_node', 'connects_to', 'sensor_node')].edge_index
            batch_edge_indices = []
            for b in range(batch_size_t):
                batch_edge_indices.append(edge_index + b * num_nodes_t)
            batch_edge_index = torch.cat(batch_edge_indices, dim=1)
            
            # 通过GCN层
            h = x
            for gcn_layer in self.gcn_layers:
                h = gcn_layer(h, batch_edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            # 重新整形为批次格式
            encoded = h.view(batch_size_t, num_nodes_t, self.hidden_dim)
            temporal_encodings.append(encoded)
        
        # 准备GRU输入 - 将时间维度放在正确位置
        # 堆叠所有时间步：[batch_size, num_nodes, window_size, hidden_dim]
        node_temporal = torch.stack(temporal_encodings, dim=2)
        # 重新整形为GRU输入格式：[batch_size * num_nodes, window_size, hidden_dim]
        gru_input = node_temporal.view(-1, window_size, self.hidden_dim)
        
        # 通过GRU层获得时序编码
        gru_out, _ = self.node_grus['sensor_node'](gru_input)
        # 取所有时间步的输出：[batch_size * num_nodes, window_size, gru_hidden_dim]
        
        # 解码重构特征
        reconstructions = {}
        
        # 重新整形：[batch_size, num_nodes, window_size, gru_hidden_dim]
        gru_output = gru_out.view(batch_size, num_nodes, window_size, self.gru_hidden_dim)
        
        # 对每个解码器生成重构结果
        for decoder_name, decoder in self.node_decoders.items():
            # 应用解码器到所有时间步
            decoder_input = gru_output.reshape(-1, self.gru_hidden_dim)  # [batch*nodes*time, gru_hidden_dim]
            decoded = decoder(decoder_input)
            
            # 重新整形到原始格式
            output_dim = decoded.size(-1)
            decoded = decoded.view(batch_size, num_nodes, window_size, output_dim)
            
            reconstructions[f'sensor_node_{decoder_name}'] = decoded
        
        return reconstructions
    
    def compute_loss(self, 
                     reconstructions: Dict[str, torch.Tensor], 
                     targets: HeteroData,
                     loss_weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算重构损失
        
        Args:
            reconstructions: 重构结果
            targets: 目标数据
            loss_weights: 损失权重
            
        Returns:
            总损失和各项损失详情
        """
        if loss_weights is None:
            loss_weights = {
                'pressure': 1.0,
                'mass_flow': 1.0,
                'temperature': 1.0,
                'derivatives': 0.5,
                'masks': 0.3,
                'states': 0.2
            }
        
        losses = {}
        total_loss = 0.0
        
        # 从目标数据提取特征
        target_features = targets['sensor_node'].x  # Should be [batch_size * num_nodes, window_size, features] or similar
        
        # Handle different tensor formats
        if len(target_features.shape) == 3:
            # Format: [batch_size * num_nodes, window_size, features]
            batch_nodes, window_size, feature_dim = target_features.shape
            num_nodes = targets['sensor_node'].num_nodes
            batch_size = batch_nodes // num_nodes
            target_features = target_features.view(batch_size, num_nodes, window_size, feature_dim)
        else:
            # Format: [batch_size, num_nodes, window_size, features] 
            batch_size, num_nodes, window_size, feature_dim = target_features.shape
        
        # 假设特征顺序是：[Pressure, Mass_Flow, Temperature, dP_dt, dT_dt, dM_dt, 
        #                   pressure_mask, flow_mask, temp_mask, state_normal, state_anomaly, state_no_demand]
        
        # 物理量损失
        if 'sensor_node_pressure' in reconstructions:
            pressure_target = target_features[:, :, :, 0:1]  # 第一个特征是压力
            pressure_pred = reconstructions['sensor_node_pressure']
            losses['pressure'] = F.mse_loss(pressure_pred, pressure_target)
            total_loss += loss_weights['pressure'] * losses['pressure']
        
        if 'sensor_node_mass_flow' in reconstructions:
            flow_target = target_features[:, :, :, 1:2]  # 第二个特征是质量流量
            flow_pred = reconstructions['sensor_node_mass_flow']
            losses['mass_flow'] = F.mse_loss(flow_pred, flow_target)
            total_loss += loss_weights['mass_flow'] * losses['mass_flow']
        
        if 'sensor_node_temperature' in reconstructions:
            temp_target = target_features[:, :, :, 2:3]  # 第三个特征是温度
            temp_pred = reconstructions['sensor_node_temperature']
            losses['temperature'] = F.mse_loss(temp_pred, temp_target)
            total_loss += loss_weights['temperature'] * losses['temperature']
        
        # 导数损失
        if 'sensor_node_derivatives' in reconstructions:
            deriv_target = target_features[:, :, :, 3:6]  # dP/dt, dT/dt, dM/dt
            deriv_pred = reconstructions['sensor_node_derivatives']
            # 对导数使用掩码损失（忽略NaN值）
            mask = ~torch.isnan(deriv_target)
            if mask.sum() > 0:
                losses['derivatives'] = F.mse_loss(deriv_pred[mask], deriv_target[mask])
                total_loss += loss_weights['derivatives'] * losses['derivatives']
        
        # 掩码损失（二值化交叉熵）
        if 'sensor_node_masks' in reconstructions:
            mask_target = target_features[:, :, :, 6:9]  # pressure_mask, flow_mask, temp_mask
            mask_pred = reconstructions['sensor_node_masks']
            losses['masks'] = F.binary_cross_entropy_with_logits(mask_pred, mask_target)
            total_loss += loss_weights['masks'] * losses['masks']
        
        # 状态损失（多分类交叉熵）
        if 'sensor_node_states' in reconstructions:
            state_target = target_features[:, :, :, 9:12]  # state_normal, state_anomaly, state_no_demand
            state_pred = reconstructions['sensor_node_states']
            # 将one-hot目标转换为类别标签
            state_labels = torch.argmax(state_target, dim=-1)
            losses['states'] = F.cross_entropy(state_pred.view(-1, 3), state_labels.view(-1))
            total_loss += loss_weights['states'] * losses['states']
        
        return total_loss, losses


def create_model(feature_dims: Dict[str, int], **kwargs) -> SteamNet_Autoencoder:
    """
    创建SteamNet模型实例
    
    Args:
        feature_dims: 特征维度字典
        **kwargs: 其他模型参数
        
    Returns:
        SteamNet_Autoencoder实例
    """
    model = SteamNet_Autoencoder(
        node_feature_dim=feature_dims['node_feature_dim'],
        edge_feature_dim=feature_dims['edge_feature_dim'],
        window_size=feature_dims['window_size'],
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("Testing SteamNet_Autoencoder...")
    
    # 模拟特征维度
    feature_dims = {
        'node_feature_dim': 12,
        'edge_feature_dim': 3,
        'window_size': 24,
        'num_nodes': 35,
        'num_edges': 25
    }
    
    # 创建模型
    model = create_model(feature_dims, hidden_dim=64, gru_hidden_dim=32)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建模拟数据
    batch_size = 2
    num_nodes = feature_dims['num_nodes']
    window_size = feature_dims['window_size']
    node_feature_dim = feature_dims['node_feature_dim']
    
    # 模拟HeteroData
    data = HeteroData()
    data['sensor_node'].x = torch.randn(batch_size, num_nodes, window_size, node_feature_dim)
    data['sensor_node'].num_nodes = num_nodes
    
    # 模拟边连接
    edge_index = torch.randint(0, num_nodes, (2, 20))  # 20条边
    data[('sensor_node', 'connects_to', 'sensor_node')].edge_index = edge_index
    
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        reconstructions = model(data)
        
    print("Forward pass successful!")
    print("Reconstruction keys:", list(reconstructions.keys()))
    for key, tensor in reconstructions.items():
        print(f"  {key}: {tensor.shape}")
    
    # 损失计算测试
    total_loss, losses = model.compute_loss(reconstructions, data)
    print(f"Total loss: {total_loss.item():.4f}")
    print("Individual losses:", {k: v.item() for k, v in losses.items()})