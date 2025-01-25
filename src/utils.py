import torch
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

def setup_logging(config=None):
    """设置日志系统
    
    Args:
        config: 日志配置信息
    """
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/run.log'
        }
    
    # 创建日志目录
    log_path = Path(config['file'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果日志文件已存在，先删除它
    if log_path.exists():
        log_path.unlink()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, config['level']),
        format=config['format'],
        handlers=[
            # 文件处理器使用UTF-8编码
            logging.FileHandler(str(log_path), encoding='utf-8', mode='w'),
            # 控制台处理器也使用UTF-8编码
            logging.StreamHandler()
        ]
    )
    
    # 设置所有处理器的编码为UTF-8
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(handler.stream)
            handler.stream.reconfigure(encoding='utf-8')

def save_model(model, path):
    """保存模型和相关状态
    
    Args:
        model: 模型实例
        path: 保存路径
    """
    # 创建保存目录
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存模型状态
    torch.save({
        'model_state_dict': model.feature_encoder.state_dict(),
        'knowledge_base': model.knowledge_base,
        'threshold': model.threshold,
        'timestamp': datetime.now().isoformat()
    }, path)
    
    logging.info(f"模型已保存到: {path}")

def load_model(model, path):
    """加载模型
    
    Args:
        model: 模型实例
        path: 模型路径
    
    Returns:
        loaded_model: 加载后的模型
    """
    try:
        checkpoint = torch.load(path)
        model.feature_encoder.load_state_dict(checkpoint['model_state_dict'])
        model.knowledge_base = checkpoint['knowledge_base']
        model.threshold = checkpoint['threshold']
        logging.info(f"模型已从 {path} 加载")
        return model
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        raise

class NumpyEncoder(json.JSONEncoder):
    """处理NumPy数据类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_results(results, path):
    """保存测试结果
    
    Args:
        results: 测试结果字典
        path: 保存路径
    """
    # 创建目录
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # 使用自定义编码器保存JSON
    with open(path, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    logging.info(f"结果已保存到: {path}")

def calculate_metrics(y_true, y_pred, is_normal):
    """计算评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        is_normal: 正常样本标记
        
    Returns:
        metrics: 评估指标字典
    """
    metrics = {
        'accuracy': np.mean(y_pred == y_true),
        'normal_rate': np.mean(is_normal),
        'unknown_rate': np.mean(~is_normal),
    }
    return metrics 