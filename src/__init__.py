"""
加密流量异常检测系统
~~~~~~~~~~~~~~~~~~~~~

这个包实现了一个基于改进孤立森林算法的加密流量异常检测系统。

主要模块:
- feature_extraction: 特征提取模块
- isolation_forest: 核心算法模块
- config: 配置模块
- utils: 工具函数模块
- test: 测试评估模块
"""

from .feature_extraction import extract_statistical_features
from .isolation_forest import EnhancedIsolationForest, FeatureEncoder
from .config import CONFIG
from .utils import (
    setup_logging,
    save_model,
    load_model,
    save_results,
    calculate_metrics
)
from .test import evaluate_model

__version__ = '1.0.0'
__author__ = '[Guojie Wang]'
__email__ = '[1934116641@qq.com]'

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'micro': 0,
    'releaselevel': 'final',
    'serial': 0
}

# 导出的主要接口
__all__ = [
    'extract_statistical_features',
    'EnhancedIsolationForest',
    'FeatureEncoder',
    'CONFIG',
    'setup_logging',
    'save_model',
    'load_model',
    'save_results',
    'calculate_metrics',
    'evaluate_model'
]

# 模块级别的文档
def get_version():
    """获取当前版本信息"""
    v = VERSION_INFO
    version = f"{v['major']}.{v['minor']}.{v['micro']}"
    if v['releaselevel'] != 'final':
        version = f"{version}-{v['releaselevel']}"
    return version

# 包的基本信息
PACKAGE_INFO = {
    'name': '加密流量异常检测系统',
    'version': get_version(),
    'description': '基于改进孤立森林算法的加密流量异常检测系统',
    'author': __author__,
    'email': __email__,
    'features': [
        '统计特征提取',
        '深度学习特征编码',
        '改进型孤立森林',
        '动态知识库更新'
    ]
}

# 初始化日志
# 从utils模块导入setup_logging函数用于初始化日志配置
# setup_logging函数负责设置日志级别、格式和输出位置等
from .utils import setup_logging
setup_logging() 