import numpy as np

def extract_statistical_features(flow_data):
    """提取流量统计特征
    
    Args:
        flow_data: 流量数据，包含数据包信息
        
    Returns:
        features: 字典，包含所有统计特征
    """
    
    features = {
        # 基本统计特征
        'n_pkts': len(flow_data['packets']),  # 数据包数量
        'duration': flow_data['duration'],     # 流持续时间
        
        # 速率特征
        'byte_rate': flow_data['bytes'] / flow_data['duration'],  # 字节率
        'pkt_rate': len(flow_data['packets']) / flow_data['duration'],  # 包率
        
        # 包间时间特征
        'iat_max': np.max(flow_data['iat']),    # 最大包间时间
        'iat_min': np.min(flow_data['iat']),    # 最小包间时间
        'iat_mean': np.mean(flow_data['iat']),  # 平均包间时间
        'iat_std': np.std(flow_data['iat']),    # 包间时间标准差
        'iat_var': np.var(flow_data['iat']),    # 包间时间方差
        
        # 包长度特征
        'pkt_len_max': np.max(flow_data['pkt_lens']),     # 最大包长
        'pkt_len_min': np.min(flow_data['pkt_lens']),     # 最小包长
        'pkt_len_avg': np.mean(flow_data['pkt_lens']),    # 平均包长
        'pkt_len_std': np.std(flow_data['pkt_lens']),     # 包长标准差
        'pkt_len_var': np.var(flow_data['pkt_lens']),     # 包长方差
        
        # 负载特征
        'payload_len_max': np.max(flow_data['payloads']),    # 最大负载长度
        'payload_len_min': np.min(flow_data['payloads']),    # 最小负载长度
        'payload_len_avg': np.mean(flow_data['payloads']),   # 平均负载长度
        'payload_len_std': np.std(flow_data['payloads'])     # 负载长度标准差
    }
    
    return features

def extract_features_batch(flows):
    """批量提取特征
    
    Args:
        flows: 流列表，每个流包含多个数据包
        
    Returns:
        X: 特征矩阵 (n_samples, n_features)
        feature_names: 特征名列表
    """
    features_list = []
    for flow in flows:
        features = extract_statistical_features(flow)
        features_list.append(list(features.values()))
    
    X = np.array(features_list)
    feature_names = list(features_list[0].keys())
    
    return X, feature_names

def normalize_features(X):
    """特征标准化
    
    Args:
        X: 特征矩阵
        
    Returns:
        X_norm: 标准化后的特征矩阵
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / (std + 1e-10)  # 避免除零
    return X_norm

class FeatureExtractor:
    """特征提取器类"""
    def __init__(self):
        self.feature_names = None
        self.mean = None
        self.std = None
        
    def fit_transform(self, flows):
        """提取并标准化特征
        
        Args:
            flows: 流列表
            
        Returns:
            X_norm: 标准化后的特征矩阵
        """
        # 提取特征
        X, self.feature_names = extract_features_batch(flows)
        
        # 计算统计量
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # 标准化
        X_norm = (X - self.mean) / (self.std + 1e-10)
        return X_norm
        
    def transform(self, flows):
        """使用已有统计量标准化新数据
        
        Args:
            flows: 新的流列表
            
        Returns:
            X_norm: 标准化后的特征矩阵
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("请先调用fit_transform")
            
        X, _ = extract_features_batch(flows)
        X_norm = (X - self.mean) / (self.std + 1e-10)
        return X_norm 