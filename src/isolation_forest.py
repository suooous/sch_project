import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif

class IsolationTreeNode:
    def __init__(self, size=0, feature_subset=None):
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.size = size
        self.label_counts = None
        self.dominant_label = None
        self.feature_subset = feature_subset  # 可用的特征子集

    def is_leaf(self):
        return self.left is None and self.right is None

# 分类树
class ClassificationIsolationTree:
    def __init__(self, max_height=None, feature_subset=None):
        self.max_height = max_height
        self.root = None
        self.height = 0
        self.feature_subset = feature_subset  # 树专用的特征子集
    
    # 训练分类树
    def fit(self, X, y):
        self.n_samples = len(X)
        if self.max_height is None:
            self.max_height = int(np.ceil(np.log2(self.n_samples)))
        self.root = self._grow_tree(X, y, 0)
        return self
    
    # 生长分类树
    def _grow_tree(self, X, y, current_height):
        n_samples, n_features = X.shape
        node = IsolationTreeNode(size=n_samples)
        
        # 统计标签分布
        unique_labels, label_counts = np.unique(y, return_counts=True)
        node.label_counts = dict(zip(unique_labels, label_counts))
        node.dominant_label = unique_labels[np.argmax(label_counts)]
        
        if current_height >= self.max_height or n_samples <= 1:
            return node
            
        # 确保特征索引在有效范围内
        available_features = (self.feature_subset if self.feature_subset is not None 
                            else range(n_features))
        available_features = [f for f in available_features if f < n_features]
        
        if not available_features:
            return node
            
        # 随机选择分割特征
        split_feature = np.random.choice(available_features)
        X_feature = X[:, split_feature]
        
        min_value = np.min(X_feature)
        max_value = np.max(X_feature)
        
        if min_value == max_value:
            return node
            
        split_value = min_value + np.random.random() * (max_value - min_value)
        
        # 分割数据
        left_mask = X_feature < split_value
        right_mask = ~left_mask
        
        node.split_feature = split_feature
        node.split_value = split_value
        
        node.left = self._grow_tree(X[left_mask], y[left_mask], current_height + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], current_height + 1)
        
        return node

    # 计算路径长度
    def path_length(self, X):
        if isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                return self._single_path_length(X)
            else:
                return np.array([self._single_path_length(x) for x in X])
        else:
            raise ValueError("Input X should be a numpy array")

    # 计算单个样本的路径长度
    def _single_path_length(self, x):
        """计算单个样本的路径长度"""
        current_node = self.root
        path_length = 0
        
        while current_node is not None:
            if current_node.is_leaf():
                break
                
            path_length += 1
            # 确保使用正确的特征子集
            feature_idx = current_node.split_feature
            if self.feature_subset is not None:
                if feature_idx >= len(x):  # 如果特征索引超出范围
                    break
            
            if x[feature_idx] < current_node.split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
                
        return path_length + self._c(current_node.size)


    def _c(self, n):
        if n <= 1:
            return 0
        # 计算路径长度 
        
        h = np.log(n - 1) + 0.5772156649
        return 2 * h - (2 * (n - 1) / n)

class EnhancedIsolationForest:
    def __init__(self, contamination=0.1, random_state=42):
        self.n_estimators = 100
        self.contamination = contamination
        self.random_state = random_state
        self.trees = []
        self.knowledge_base = {
            'class_stats': {},
            'thresholds': {},
            'feature_importance': {}
        }
        self.threshold_ = None  # 添加阈值属性

    def _select_features(self, X, y):
        """精简的特征选择
        从20个属性中选择6-8个最重要的特征"""
        unique_classes = np.unique(y)
        feature_importance = {}
        
        for label in unique_classes:
            y_binary = (y == label).astype(int)
            importance_scores = mutual_info_classif(X, y_binary)
            
            # 选择前7个最重要的特征
            n_select = 7
            top_features = np.argsort(importance_scores)[-n_select:]
            
            feature_importance[label] = {
                'scores': importance_scores,
                'selected_features': top_features
            }
        
        return feature_importance

    def fit(self, X, y):
        np.random.seed(self.random_state)
        
        # 特征选择
        feature_importance = self._select_features(X, y)
        
        # 为每个类别构建专门的树
        unique_classes = np.unique(y)
        trees_per_class = self.n_estimators // len(unique_classes)
        
        for label in unique_classes:
            features = feature_importance[label]['selected_features']
            mask = y == label
            X_class = X[mask]
            y_class = y[mask]
            
            for _ in range(trees_per_class):
                tree = ClassificationIsolationTree(feature_subset=features)
                tree.fit(X_class, y_class)
                self.trees.append((label, tree))
        
        # 计算阈值
        scores = self.score_samples(X)
        self.threshold_ = np.percentile(scores, (1 - self.contamination) * 100)
        
        self._build_knowledge_base(X, y)
        return self

    def _build_knowledge_base(self, X, y):
        """改进的知识库构建"""
        unique_classes = np.unique(y)
        class_scores = {}
        
        # 收集每个类别的分数分布
        for label in unique_classes:
            mask = y == label
            X_class = X[mask]
            scores = self.score_samples(X_class)
            class_scores[label] = scores
            
            # 存储统计信息
            self.knowledge_base['class_stats'][label] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'sample_count': len(X_class)
            }
        
        # 分析类别间的分数重叠
        overlap_matrix = {}
        for label1 in unique_classes:
            overlap_matrix[label1] = {}
            scores1 = class_scores[label1]
            for label2 in unique_classes:
                if label1 != label2:
                    scores2 = class_scores[label2]
                    overlap = np.sum((scores2 >= np.min(scores1)) & 
                                   (scores2 <= np.max(scores1))) / len(scores2)
                    overlap_matrix[label1][label2] = overlap
        
        self.knowledge_base['overlap_matrix'] = overlap_matrix
        
        # 智能设置阈值
        for label in unique_classes:
            scores = class_scores[label]
            total_overlap = sum(overlap_matrix[label].values())
            
            # 根据重叠程度调整阈值百分位数
            base_percentile = 95
            overlap_adjustment = total_overlap * 2
            final_percentile = max(base_percentile - overlap_adjustment, 85)
            
            self.knowledge_base['thresholds'][label] = np.percentile(scores, final_percentile)

    def score_samples(self, X):
        """计算样本的异常分数"""
        scores = np.zeros(len(X))
        tree_counts = np.zeros(len(X))
        
        for label, tree in self.trees:
            # 确保使用正确的特征索引
            if hasattr(tree, 'feature_subset') and tree.feature_subset is not None:
                # 检查特征索引是否有效
                valid_features = [f for f in tree.feature_subset if f < X.shape[1]]
                if valid_features:
                    X_subset = X[:, valid_features]
                else:
                    continue
            else:
                X_subset = X
            
            tree_scores = tree.path_length(X_subset)
            scores += tree_scores
            tree_counts += 1
        
        # 计算平均分数
        scores = scores / np.maximum(tree_counts, 1)
        
        # 标准化分数
        n_samples = len(X)
        average_path_length = self._average_path_length(n_samples)
        scores = 2 ** (-scores / average_path_length)
        
        return scores

    def _average_path_length(self, n_samples):
        if n_samples <= 1:
            return 1
        h = np.log(n_samples - 1) + 0.5772156649
        return 2 * h - (2 * (n_samples - 1) / n_samples)

    def predict(self, X):
        """改进的预测方法"""
        scores = self.score_samples(X)
        predictions = []
        
        for score in scores:
            best_fit = float('-inf')
            best_class = None
            
            # 对每个类别计算匹配度
            for label, stats in self.knowledge_base['class_stats'].items():
                mean_score = stats['mean_score']
                std_score = stats['std_score']
                
                # 使用高斯核计算匹配度
                if std_score > 0:
                    z_score = abs(score - mean_score) / std_score
                    fit = np.exp(-0.5 * z_score ** 2)
                else:
                    fit = 1.0 if score == mean_score else 0.0
                
                if fit > best_fit:
                    best_fit = fit
                    best_class = label
            
            predictions.append(best_class)
        
        return np.array(predictions)

    def predict_and_classify(self, X):
        """
        只返回是否异常的预测结果，不再进行具体分类
        """
        # 获取异常分数
        scores = self.score_samples(X)
        # 根据阈值判断是否异常
        is_normal = scores >= self.threshold_
        return is_normal, scores 