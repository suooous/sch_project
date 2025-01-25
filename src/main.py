import logging
import os
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scenario1_known_traffic import evaluate_scenario1, load_and_preprocess_data

# Remove imports for scenarios 2 and 3
# from scenario2_unknown_traffic import evaluate_scenario2
# from scenario3_adversarial import evaluate_scenario3

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def save_results(results, output_dir='results'):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果到JSON文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, 'test_results.json')
    
    # 写入结果
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logger.info(f'结果已保存到: {filepath}')

def main():
    logger.info('加载训练数据...')
    X_train, y_train = load_and_preprocess_data('data/processed/known_traffic.csv', logger)
    
    logger.info('开始训练增强型孤立森林模型...')
    
    logger.info('加载测试数据...')
    X_test, y_test = load_and_preprocess_data('data/processed/known_test.csv', logger)
    
    logger.info('进行预测和分类...')
    
    results = evaluate_scenario1((X_train, y_train), (X_test, y_test))
    
    logger.info('\n评估结果:')
    logger.info(f"准确率: {results['classification_accuracy']:.4f}")
    logger.info(f"正常样本比例: {results['normal_ratio']:.4f}")
    logger.info('\n分类报告:')
    if 'normal_samples_metrics' in results:
        logger.info(results['normal_samples_metrics']['classification_report'])
    
    save_results(results)

if __name__ == '__main__':
    main() 
    