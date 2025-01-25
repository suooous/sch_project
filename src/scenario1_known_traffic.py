import pandas as pd
import numpy as np
import logging
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from isolation_forest import EnhancedIsolationForest
from utils import setup_logging, save_results
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from datetime import datetime

# 设置matplotlib的全局配置
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_and_preprocess_data(csv_path, logger=None):
    """改进的数据加载和预处理"""
    df = pd.read_csv(csv_path)
    
    # 1. 基础特征提取
    X = df.drop(['Label', 'flow_key'], axis=1)
    y = df['Label']
    
    # 2. 特征工程
    X = add_engineered_features(X)
    
    # 3. 特征标准化 - 使用稳健的方法
    scaler = RobustScaler(quantile_range=(5, 95))
    X_processed = scaler.fit_transform(X)
    
    # 4. 添加有限的多项式特征（只对最重要的特征）
    important_features = ['duration', 'pkt_rate', 'byte_rate']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    important_indices = [X.columns.get_loc(col) for col in important_features if col in X.columns]
    if important_indices:
        X_poly = poly.fit_transform(X_processed[:, important_indices])
        X_processed = np.hstack([X_processed, X_poly[:, len(important_indices)+1:]])
    
    if logger:
        logger.info(f"\n数据加载完成:")
        logger.info(f"原始特征数量: {X.shape[1]}")
        logger.info(f"处理后特征数量: {X_processed.shape[1]}")
        logger.info(f"样本数量: {X_processed.shape[0]}")
        logger.info("\n类别分布:")
        for label, count in y.value_counts().items():
            logger.info(f"{label}: {count}")
    
    return X_processed, y.values

def add_engineered_features(X):
    """优化后的特征工程，专注于最有意义的特征"""
    X_new = pd.DataFrame()
    
    # 首先打印列名以便调试
    logger = logging.getLogger(__name__)
    logger.info("可用的特征列: " + ", ".join(X.columns))
    
    # 1. 时间特征
    X_new['duration'] = X['duration']
    X_new['iat_mean'] = X['iat_mean']
    X_new['iat_std'] = X['iat_std']
    
    # 2. 数据包特征
    X_new['n_pkts'] = X['n_pkts']
    X_new['pkt_rate'] = X['pkt_rate']
    X_new['byte_rate'] = X['byte_rate']
    
    # 3. 包长特征 - 使用实际的列名
    X_new['pkt_len_mean'] = X['pkt_len_avg']  # 改用 pkt_len_avg
    X_new['pkt_len_std'] = X['pkt_len_std']
    X_new['payload_len_mean'] = X['payload_len_avg']  # 改用 payload_len_avg
    X_new['payload_len_std'] = X['payload_len_std']
    
    # 4. 比率特征
    X_new['bytes_per_packet'] = X['byte_rate'] / (X['pkt_rate'] + 1e-6)
    X_new['payload_ratio'] = X['payload_len_avg'] / (X['pkt_len_avg'] + 1e-6)  # 使用正确的列名
    
    # 5. 高级统计特征
    if 'pkt_len_skew' in X.columns:
        X_new['pkt_len_skew'] = X['pkt_len_skew']
    if 'iat_skew' in X.columns:
        X_new['iat_skew'] = X['iat_skew']
    
    return X_new

def check_data_quality(df, dataset_name):
    """检查数据质量并打印报告"""
    logger = logging.getLogger(__name__)
    logger.info(f"\n检查数据集: {dataset_name}")
    
    # 1. 详细的类别分布
    class_dist = df['Label'].value_counts()
    logger.info("\n类别分布:")
    for label, count in class_dist.items():
        logger.info(f"{label}: {count} ({count/len(df)*100:.2f}%)")
    
    # 2. 特征相关性分析
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    logger.info("\n高相关性特征对:")
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            if abs(corr_matrix.iloc[i,j]) > 0.8:
                logger.info(f"{numeric_cols[i]} - {numeric_cols[j]}: {corr_matrix.iloc[i,j]:.3f}")

def evaluate_known_traffic(y_true, y_pred, sample_weights, anomaly_scores):
    """增强的评估函数"""
    results = {
        'classification_accuracy': np.mean(y_true == y_pred),
        'normal_ratio': np.mean(sample_weights),
        'anomaly_score_stats': {
            'mean': np.mean(anomaly_scores),
            'std': np.std(anomaly_scores),
            'min': np.min(anomaly_scores),
            'max': np.max(anomaly_scores),
            'percentiles': {
                '25': np.percentile(anomaly_scores, 25),
                '50': np.percentile(anomaly_scores, 50),
                '75': np.percentile(anomaly_scores, 75),
                '90': np.percentile(anomaly_scores, 90)
            }
        }
    }
    
    # 只对正常样本计算分类指标
    normal_mask = sample_weights == 1
    if np.any(normal_mask):
        y_true_normal = y_true[normal_mask]
        y_pred_normal = y_pred[normal_mask]
        
        results.update({
            'normal_samples_metrics': {
                'classification_report': classification_report(
                    y_true_normal, 
                    y_pred_normal, 
                    zero_division=0
                ),
                'confusion_matrix': confusion_matrix(
                    y_true_normal, 
                    y_pred_normal
                ).tolist()
            }
        })
    
    return results

def visualize_known_traffic_results(results, save_dir='results/scenario1/visualize_known_traffic_results'):
    """可视化已知类别流量的测试结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用与结果文件相同的时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 设置保存参数
    save_params = {
        'bbox_inches': 'tight',
        'format': 'png'  # 明确指定格式
    }
    
    # 1. 绘制异常分数分布
    plt.figure(figsize=(10, 6))
    plt.hist(results['anomaly_scores'], bins=50)
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')

    # 创建异常分数保存目录
    save_dir_anomaly_scores = os.path.join(save_dir, f'anomaly_scores_distribution')
    os.makedirs(save_dir_anomaly_scores, exist_ok=True)

    plt.savefig(os.path.join(save_dir_anomaly_scores, f'anomaly_scores_distribution_{timestamp}.png'), **save_params)
    plt.close()
    
    
    # 2. 绘制混淆矩阵
    plt.figure(figsize=(12, 8))
    unique_labels = sorted(list(set(results['true_labels'])))
    sns.heatmap(results['confusion_matrix'], 
                annot=True, 
                fmt='d',
                xticklabels=unique_labels,
                yticklabels=unique_labels)

    save_dir_confusion_matrix = os.path.join(save_dir, f'confusion_matrix')
    os.makedirs(save_dir_confusion_matrix, exist_ok=True)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir_confusion_matrix, f'confusion_matrix_{timestamp}.png'), **save_params)
    plt.close()
    
    # 3. 绘制每个类别的准确率
    plt.figure(figsize=(10, 6))
    accuracies = results['per_class_accuracy']
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title('Accuracy per Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_dir_per_class_accuracy = os.path.join(save_dir, f'per_class_accuracy')
    os.makedirs(save_dir_per_class_accuracy, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir_per_class_accuracy, f'per_class_accuracy_{timestamp}.png'), **save_params)
    plt.close()

def setup_logging():
    """设置日志记录"""
    from datetime import datetime
    
    # 创建logs目录
    log_dir = 'logs/scenario1'
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'run_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return log_file

class HybridDetector:
    def __init__(self, contamination=0.1, random_state=42):
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            max_features=1.0,
            bootstrap=False,
            n_jobs=None,
            random_state=random_state,
            verbose=0,
            warm_start=False
        )
        
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        
        # 选择10个最重要的特征
        self.feature_indices = [
            0,  # n_pkts - 流中的数据包数量
            1,  # duration - 流持续时间
            2,  # byte_rate - 字节率
            3,  # pkt_rate - 数据包率
            6,  # iat_mean - 数据包间平均到达时间
            7,  # iat_std - 数据包间到达时间标准差
            11, # pkt_len_avg - 平均包长度
            12, # pkt_len_std - 包长度标准差
            9,  # pkt_len_max - 最大包长度
            10  # pkt_len_min - 最小包长度
        ]
        
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X_selected = X[:, self.feature_indices]
        else:
            X_selected = np.array(X)[:, self.feature_indices]
        
        # 训练隔离森林
        self.isolation_forest.fit(X_selected)
        
        # 训练随机森林
        self.random_forest.fit(X_selected, y)
        
        return self
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X_selected = X[:, self.feature_indices]
        else:
            X_selected = np.array(X)[:, self.feature_indices]
        
        # 获取隔离森林的决策分数（而不是原始分数）
        anomaly_scores = self.isolation_forest.decision_function(X_selected)
        
        # 获取随机森林的预测结果和概率
        rf_predictions = self.random_forest.predict(X_selected)
        rf_proba = self.random_forest.predict_proba(X_selected)
        
        # 结合两个模型的结果
        predictions = []
        for i, (score, rf_pred, probs) in enumerate(zip(anomaly_scores, rf_predictions, rf_proba)):
            # 使用决策函数的值判断是否为异常
            # 如果score < 0，则被认为是异常
            if score < 0 and np.max(probs) < 0.6:
                predictions.append("Unknown")
            else:
                predictions.append(rf_pred)
                
        return np.array(predictions)

    def get_anomaly_scores(self, X):
        """获取异常分数"""
        if isinstance(X, np.ndarray):
            X_selected = X[:, self.feature_indices]
        else:
            X_selected = np.array(X)[:, self.feature_indices]
        return -self.isolation_forest.decision_function(X_selected)

def evaluate_scenario1(train_data, test_data):
    """评估已知流量场景"""
    # 解包数据
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # 确保标签是字符串类型
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    
    # 训练和预测
    detector = HybridDetector(contamination=0.1)
    detector.fit(X_train, y_train)
    y_pred = detector.predict(X_test)
    
    # 获取异常分数
    anomaly_scores = detector.get_anomaly_scores(X_test)
    
    # 计算每个类别的准确率
    unique_labels = np.unique(y_test)
    per_class_accuracy = {
        label: np.mean(y_pred[y_test == label] == label)
        for label in unique_labels
    }
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 构建结果字典
    results = evaluate_known_traffic(
        y_test, 
        y_pred, 
        np.ones(len(y_test)), 
        anomaly_scores
    )
    
    # 添加可视化所需的额外数据
    results.update({
        'anomaly_scores': anomaly_scores,
        'true_labels': y_test,
        'predicted_labels': y_pred,
        'confusion_matrix': conf_matrix,
        'per_class_accuracy': per_class_accuracy
    })
    
    return results

def main():
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. 加载和预处理数据
        logger.info("加载训练数据...")
        X_train, y_train = load_and_preprocess_data('data/processed/known_traffic.csv', logger)
        
        # 2. 训练模型
        logger.info("\n开始训练混合检测模型...")
        model = HybridDetector(contamination=0.1)  # 增加contamination值
        model.fit(X_train, y_train)
        
        # 3. 加载测试数据
        logger.info("\n加载测试数据...")
        X_test, y_test = load_and_preprocess_data('data/processed/known_test.csv', logger)
        
        # 4. 预测
        logger.info("\n进行预测...")
        predicted_labels = model.predict(X_test)
        
        # 5. 评估结果
        results = evaluate_scenario1(
            (X_train, y_train),
            (X_test, y_test)
        )
        
        # 6. 输出详细的评估信息
        logger.info("\n评估结果:")
        logger.info(f"整体准确率: {results['classification_accuracy']:.4f}")
        logger.info(f"正常样本比例: {results['normal_ratio']:.4f}")
        logger.info("\n异常分数统计:")
        for key, value in results['anomaly_score_stats'].items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"{key}: {value:.4f}")
        
        if 'normal_samples_metrics' in results:
            logger.info("\n正常样本分类报告:")
            logger.info(results['normal_samples_metrics']['classification_report'])
        
        
        from datetime import datetime
        # 7. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join('results/scenario1/results', f'test_results_{timestamp}.json')
        save_results(results, results_file)
        
        # 8. 生成可视化结果
        logger.info("\n生成可视化结果...")
        visualize_known_traffic_results(results)
        
    except Exception as e:
        logger.error(f"运行过程中出现错误: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 