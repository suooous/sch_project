import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, scaler):
    """评估模型性能"""
    # 标准化测试数据
    X_normalized = scaler.transform(X_test)
    
    # 预测
    is_normal, predicted_labels = model.predict_and_classify(X_normalized)
    
    # 计算指标
    results = {
        'accuracy': np.mean(predicted_labels == y_test),
        'normal_detection_rate': np.mean(is_normal),
        'unknown_detection_rate': np.mean(~is_normal),
        'classification_report': classification_report(
            y_test[is_normal], 
            predicted_labels[is_normal]
        )
    }
    
    return results 