"""
Stage 1: Data Preprocessing & Feature Engineering
构建SteamNet项目的数据预处理与特征工程脚本
"""

import json
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple, Any


def load_and_merge_timeseries_data(json_path: str, csv_folder_path: str) -> pd.DataFrame:
    """
    加载原始数据：读取蓬莱蒸汽S40.json中的nodelist，并合并所有CSV数据
    
    Args:
        json_path: 拓扑结构JSON文件路径
        csv_folder_path: CSV数据文件夹路径
    
    Returns:
        合并后的DataFrame，使用node_id和timestamp作为多级索引
    """
    print("Loading topology structure...")
    with open(json_path, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)
    
    nodelist = topology_data['nodelist']
    print(f"Found {len(nodelist)} nodes in topology")
    
    # 收集所有CSV数据
    all_dataframes = []
    csv_folder = Path(csv_folder_path)
    
    for node in nodelist:
        node_name = node['name']
        csv_file = csv_folder / f"{node_name}.csv"
        
        if csv_file.exists():
            print(f"Loading data for node: {node_name}")
            df = pd.read_csv(csv_file)
            df['Node_ID'] = node_name
            all_dataframes.append(df)
        else:
            print(f"Warning: CSV file not found for node: {node_name}")
    
    if not all_dataframes:
        raise ValueError("No CSV files found matching node names")
    
    # 合并所有数据
    print("Merging all timeseries data...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 转换timestamp为datetime
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    
    # 设置多级索引
    merged_df = merged_df.set_index(['Node_ID', 'timestamp'])
    merged_df = merged_df.sort_index()
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Time range: {merged_df.index.get_level_values('timestamp').min()} to {merged_df.index.get_level_values('timestamp').max()}")
    
    return merged_df


def build_static_graph(json_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    静态图结构构建：创建节点和边的映射，构建异构图的edge_index_dict
    
    Args:
        json_path: 拓扑结构JSON文件路径
    
    Returns:
        node_map, link_map, edge_index_dict
    """
    print("Building static graph structure...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)
    
    nodelist = topology_data['nodelist']
    linklist = topology_data['linklist']
    
    # 创建节点映射
    node_map = {}
    for idx, node in enumerate(nodelist):
        node_name = node['name']
        node_type = node.get('parameter', {}).get('type', 'sensor_node')  # 默认类型
        node_map[node_name] = {
            'idx': idx,
            'type': node_type,
            'name': node_name
        }
    
    # 创建边映射
    link_map = {}
    for idx, link in enumerate(linklist):
        link_name = link['name']
        link_type = link.get('parameter', {}).get('type', 'pipe')  # 默认类型
        link_map[link_name] = {
            'idx': idx,
            'type': link_type,
            'name': link_name,
            'source': link.get('source', ''),
            'target': link.get('target', '')
        }
    
    # 构建异构图的edge_index_dict
    # 这里简化处理，假设所有节点连接都是相同类型
    edge_index_dict = {}
    
    # 收集边连接信息
    edges = []
    for link in linklist:
        source = link.get('source', '')
        target = link.get('target', '')
        
        if source in node_map and target in node_map:
            source_idx = node_map[source]['idx']
            target_idx = node_map[target]['idx']
            edges.append([source_idx, target_idx])
    
    if edges:
        edge_index = np.array(edges).T
        # 简化：所有连接都作为sensor_node到sensor_node的连接
        edge_index_dict[('sensor_node', 'connects_to', 'sensor_node')] = edge_index
    
    print(f"Created {len(node_map)} nodes and {len(link_map)} links")
    print(f"Edge index dict keys: {list(edge_index_dict.keys())}")
    
    return node_map, link_map, edge_index_dict


def create_static_edge_features(json_path: str, link_map: Dict) -> Tuple[np.ndarray, StandardScaler]:
    """
    静态边特征提取：从linklist中提取物理属性并归一化
    
    Args:
        json_path: 拓扑结构JSON文件路径
        link_map: 边映射字典
    
    Returns:
        归一化后的特征矩阵和定标器对象
    """
    print("Creating static edge features...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)
    
    linklist = topology_data['linklist']
    
    # 提取边特征
    features = []
    for link in linklist:
        param = link.get('parameter', {})
        
        # 提取物理属性，如果不存在则使用默认值
        length = param.get('Length', 100.0)  # 默认长度
        inner_diameter = param.get('Inner_Diameter', 0.5)  # 默认内径
        roughness = param.get('Roughness', 0.0001)  # 默认粗糙度
        
        features.append([length, inner_diameter, roughness])
    
    features = np.array(features)
    
    # 归一化
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    print(f"Created static edge features with shape: {normalized_features.shape}")
    
    return normalized_features, scaler


def create_dynamic_features(merged_df: pd.DataFrame, node_map: Dict, adjacency_dict: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    动态特征工程与标注：创建节点特征、边特征和标签
    
    Args:
        merged_df: 合并后的时序数据
        node_map: 节点映射字典
        adjacency_dict: 邻接关系字典
    
    Returns:
        节点特征DataFrame和边特征DataFrame
    """
    print("Creating dynamic features and labels...")
    
    # 重置索引以便操作
    df = merged_df.reset_index()
    
    # 1. 标注：创建label列
    # 简化版本：根据数据异常程度进行标注
    df['label'] = 0  # 默认为正常
    
    # 检测异常值（这里使用简单的统计方法，实际应根据具体规则）
    for feature in ['Pressure', 'Mass_Flow', 'Temperature']:
        if feature in df.columns:
            # 计算z-score
            z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
            # 标记异常值
            anomaly_mask = z_scores > 3.0
            df.loc[anomaly_mask, 'label'] = 1
    
    # 标记无需求的节点（质量流量接近0）
    if 'Mass_Flow' in df.columns:
        no_demand_mask = np.abs(df['Mass_Flow']) < 1.0
        df.loc[no_demand_mask, 'label'] = 2
    
    # 2. 节点特征：计算时间导数和其他特征
    node_features_list = []
    
    for node_id in df['Node_ID'].unique():
        node_data = df[df['Node_ID'] == node_id].copy()
        node_data = node_data.sort_values('timestamp')
        
        # 计算时间导数
        if 'Pressure' in node_data.columns:
            node_data['dP_dt'] = node_data['Pressure'].diff() / node_data['timestamp'].diff().dt.total_seconds()
        if 'Temperature' in node_data.columns:
            node_data['dT_dt'] = node_data['Temperature'].diff() / node_data['timestamp'].diff().dt.total_seconds()
        if 'Mass_Flow' in node_data.columns:
            node_data['dM_dt'] = node_data['Mass_Flow'].diff() / node_data['timestamp'].diff().dt.total_seconds()
        
        # 创建掩码向量（用于指示有效数据）
        node_data['pressure_mask'] = (~node_data['Pressure'].isna()).astype(int) if 'Pressure' in node_data.columns else 1
        node_data['flow_mask'] = (~node_data['Mass_Flow'].isna()).astype(int) if 'Mass_Flow' in node_data.columns else 1
        node_data['temp_mask'] = (~node_data['Temperature'].isna()).astype(int) if 'Temperature' in node_data.columns else 1
        
        # One-Hot状态向量（基于label）
        node_data['state_normal'] = (node_data['label'] == 0).astype(int)
        node_data['state_anomaly'] = (node_data['label'] == 1).astype(int)
        node_data['state_no_demand'] = (node_data['label'] == 2).astype(int)
        
        node_features_list.append(node_data)
    
    # 合并所有节点特征
    node_features_df = pd.concat(node_features_list, ignore_index=True)
    
    # 3. 边特征：计算动态边特征
    edge_features_list = []
    
    # 获取边连接信息
    if ('sensor_node', 'connects_to', 'sensor_node') in adjacency_dict:
        edge_index = adjacency_dict[('sensor_node', 'connects_to', 'sensor_node')]
        
        # 为每个时间点计算边特征
        timestamps = df['timestamp'].unique()
        
        for timestamp in timestamps:
            timestamp_data = df[df['timestamp'] == timestamp]
            
            for i in range(edge_index.shape[1]):
                source_idx = edge_index[0, i]
                target_idx = edge_index[1, i]
                
                # 根据索引获取节点名称
                source_node = None
                target_node = None
                for node_name, node_info in node_map.items():
                    if node_info['idx'] == source_idx:
                        source_node = node_name
                    elif node_info['idx'] == target_idx:
                        target_node = node_name
                
                if source_node and target_node:
                    source_data = timestamp_data[timestamp_data['Node_ID'] == source_node]
                    target_data = timestamp_data[timestamp_data['Node_ID'] == target_node]
                    
                    if not source_data.empty and not target_data.empty:
                        source_row = source_data.iloc[0]
                        target_row = target_data.iloc[0]
                        
                        # 计算边特征
                        pressure_drop = source_row.get('Pressure', 0) - target_row.get('Pressure', 0)
                        temperature_drop = source_row.get('Temperature', 0) - target_row.get('Temperature', 0)
                        
                        # Mode指示器（简化：基于流量）
                        avg_flow = (source_row.get('Mass_Flow', 0) + target_row.get('Mass_Flow', 0)) / 2
                        mode_indicator = 1 if avg_flow > 0 else 0
                        
                        # 质量平衡残差（简化计算）
                        mass_balance_residual = abs(source_row.get('Mass_Flow', 0) - target_row.get('Mass_Flow', 0))
                        
                        edge_features_list.append({
                            'timestamp': timestamp,
                            'edge_idx': i,
                            'source_node': source_node,
                            'target_node': target_node,
                            'Pressure_Drop': pressure_drop,
                            'Temperature_Drop': temperature_drop,
                            'Mode_Indicator': mode_indicator,
                            'Mass_Balance_Residual': mass_balance_residual
                        })
    
    edge_features_df = pd.DataFrame(edge_features_list) if edge_features_list else pd.DataFrame()
    
    print(f"Created node features with shape: {node_features_df.shape}")
    print(f"Created edge features with shape: {edge_features_df.shape}")
    print(f"Label distribution: {node_features_df['label'].value_counts().to_dict()}")
    
    return node_features_df, edge_features_df


def main():
    """主函数：按顺序调用所有函数完成数据预处理"""
    print("Starting Stage 1: Data Preprocessing & Feature Engineering")
    
    # 设置路径
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "data" / "raw" / "蓬莱蒸汽S40.json"
    csv_folder_path = base_dir / "data" / "raw" / "Sim_Dataset"
    processed_dir = base_dir / "data" / "processed"
    
    # 确保输出目录存在
    processed_dir.mkdir(exist_ok=True)
    
    try:
        # 1. 加载和合并时序数据
        merged_df = load_and_merge_timeseries_data(str(json_path), str(csv_folder_path))
        
        # 2. 构建静态图结构
        node_map, link_map, edge_index_dict = build_static_graph(str(json_path))
        
        # 保存映射文件
        with open(processed_dir / "node_map.json", 'w', encoding='utf-8') as f:
            json.dump(node_map, f, ensure_ascii=False, indent=2)
        
        with open(processed_dir / "link_map.json", 'w', encoding='utf-8') as f:
            json.dump(link_map, f, ensure_ascii=False, indent=2)
        
        with open(processed_dir / "adjacency.pkl", 'wb') as f:
            pickle.dump(edge_index_dict, f)
        
        # 3. 创建静态边特征
        static_edge_features, scaler = create_static_edge_features(str(json_path), link_map)
        
        # 保存静态边特征和定标器
        joblib.dump(scaler, processed_dir / "static_feature_scaler.joblib")
        
        static_edge_df = pd.DataFrame(static_edge_features, columns=['Length', 'Inner_Diameter', 'Roughness'])
        static_edge_df.to_parquet(processed_dir / "static_edge_features.parquet", index=False)
        
        # 4. 创建动态特征
        node_features_df, edge_features_df = create_dynamic_features(merged_df, node_map, edge_index_dict)
        
        # 保存动态特征
        node_features_df.to_parquet(processed_dir / "timeseries_node_features.parquet", index=False)
        
        if not edge_features_df.empty:
            edge_features_df.to_parquet(processed_dir / "timeseries_edge_features.parquet", index=False)
        else:
            # 创建空的边特征文件
            pd.DataFrame().to_parquet(processed_dir / "timeseries_edge_features.parquet", index=False)
        
        print("Stage 1 completed successfully!")
        print(f"Processed files saved to: {processed_dir}")
        
        # 显示处理结果摘要
        print("\n=== Processing Summary ===")
        print(f"Nodes: {len(node_map)}")
        print(f"Links: {len(link_map)}")
        print(f"Node features shape: {node_features_df.shape}")
        print(f"Edge features shape: {edge_features_df.shape}")
        print(f"Time range: {node_features_df['timestamp'].min()} to {node_features_df['timestamp'].max()}")
        
    except Exception as e:
        print(f"Error in Stage 1: {e}")
        raise


if __name__ == "__main__":
    main()