"""配置文件"""
CONFIG = {
    # 模型参数
    'model': {
        'n_estimators': 100,
        'contamination': 'auto'
    },
    
    # 数据参数
    'data': {
        'train_path': 'data/processed/normal_traffic.csv',
        'test_path': 'data/processed/mixed_traffic.csv'
    },
    
    # 特征参数
    'features': {
        'window_size': 100,
        'step_size': 50
    },
    
    # 训练参数
    'training': {
        'batch_size': 32,
        'epochs': 10
    },
    
    # 路径参数
    'paths': {
        'model_dir': 'models/saved_models',
        'log_dir': 'logs',
        'results_dir': 'results'
    }
} 