import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


class AdvancedOptimizationStrategies:
    def __init__(self):
        self.strategies = {}

    def analyze_remaining_potential(self, X_train, y_train, X_test, y_test, current_r2=0.3539):
        """分析剩余优化潜力"""
        print("分析进一步优化潜力...")

        # 1. 检查特征与目标的关系非线性
        print("\n1. 特征-目标关系分析")
        correlation_with_target = X_train.corrwith(y_train).abs()
        high_corr_features = correlation_with_target[correlation_with_target > 0.1]
        print(f"与目标相关性>0.1的特征数量: {len(high_corr_features)}")

        # 2. 检查模型复杂度是否足够
        print("\n2. 模型复杂度分析")
        test_models = [
            ('保守', {'n_estimators': 500, 'max_depth': 20}),
            ('平衡', {'n_estimators': 1000, 'max_depth': 30}),
            ('激进', {'n_estimators': 2000, 'max_depth': 50}),
            ('超激进', {'n_estimators': 3000, 'max_depth': None})
        ]

        for name, params in test_models:
            model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            print(f"  {name}模型 - CV R²: {cv_scores.mean():.4f}")

        # 3. 检查数据质量问题
        print("\n3. 数据质量分析")
        print(f"  训练集样本数: {len(X_train)}")
        print(f"  特征数量: {X_train.shape[1]}")
        print(f"  目标变量范围: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"  目标变量标准差: {y_train.std():.3f}")

        return len(high_corr_features)


class NextLevelOptimizer:
    def __init__(self):
        self.best_r2 = 0.3539

    def deep_feature_engineering(self, X, y=None):
        """深度特征工程"""
        print("进行深度特征工程...")
        X_deep = X.copy()

        # 基于领域知识的特征构造
        # 1. 聚类特征
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=5, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            X_deep['cluster'] = cluster_labels
        except:
            pass

        # 2. 主成分特征
        from sklearn.decomposition import PCA
        try:
            pca = PCA(n_components=10, random_state=42)
            pca_features = pca.fit_transform(X)
            for i in range(5):
                X_deep[f'pca_{i + 1}'] = pca_features[:, i]
        except:
            pass

        # 3. 多项式特征（选择性）
        important_features = ['feat_339', 'feat_348', 'feat_338', 'Length', 'feat_342']
        for feat in important_features:
            if feat in X.columns:
                # 高阶多项式
                X_deep[f'{feat}_cubed'] = X[feat] ** 3
                X_deep[f'{feat}_exp'] = np.exp(0.1 * np.abs(X[feat]))

        print(f"深度特征工程后维度: {X_deep.shape}")
        return X_deep

    def advanced_ensemble_architectures(self, X_train, y_train, X_test, y_test):
        """高级集成架构"""
        print("尝试高级集成架构...")

        # 1. 分层集成
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

        # 第一层：多种算法
        layer1_models = {
            'et1': ExtraTreesRegressor(n_estimators=1000, max_depth=35, random_state=1, n_jobs=-1),
            'et2': ExtraTreesRegressor(n_estimators=800, max_depth=40, random_state=2, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=500, max_depth=6, random_state=42),
            'rf': RandomForestRegressor(n_estimators=800, max_depth=30, random_state=42, n_jobs=-1)
        }

        # 训练第一层
        layer1_predictions_train = []
        layer1_predictions_test = []

        for name, model in layer1_models.items():
            print(f"  训练第一层模型: {name}")
            model.fit(X_train, y_train)
            layer1_predictions_train.append(model.predict(X_train))
            layer1_predictions_test.append(model.predict(X_test))

        # 创建第二层特征
        X_train_layer2 = np.column_stack(layer1_predictions_train)
        X_test_layer2 = np.column_stack(layer1_predictions_test)

        # 第二层：元学习器
        meta_learners = {
            'linear': ExtraTreesRegressor(n_estimators=500, max_depth=20, random_state=42),
            'nonlinear': GradientBoostingRegressor(n_estimators=300, random_state=42)
        }

        best_meta_r2 = -1
        best_meta_pred = None

        for name, meta_learner in meta_learners.items():
            meta_learner.fit(X_train_layer2, y_train)
            meta_pred = meta_learner.predict(X_test_layer2)
            meta_r2 = r2_score(y_test, meta_pred)

            print(f"  元学习器 {name}: R² = {meta_r2:.4f}")

            if meta_r2 > best_meta_r2:
                best_meta_r2 = meta_r2
                best_meta_pred = meta_pred

        return best_meta_pred, best_meta_r2

    def target_transformation(self, X_train, y_train, X_test, y_test):
        """目标变量变换"""
        print("尝试目标变量变换...")

        # 1. 对数变换
        y_train_log = np.log1p(y_train - y_train.min() + 1e-8)

        model_log = ExtraTreesRegressor(n_estimators=1000, max_depth=35, random_state=42, n_jobs=-1)
        model_log.fit(X_train, y_train_log)
        pred_log = model_log.predict(X_test)
        pred_original = np.expm1(pred_log) + y_train.min() - 1e-8
        r2_log = r2_score(y_test, pred_original)

        # 2. Box-Cox变换
        from scipy import stats
        try:
            y_train_boxcox, lambda_val = stats.boxcox(y_train - y_train.min() + 1)
            model_boxcox = ExtraTreesRegressor(n_estimators=1000, max_depth=35, random_state=42, n_jobs=-1)
            model_boxcox.fit(X_train, y_train_boxcox)
            pred_boxcox = model_boxcox.predict(X_test)
            pred_original_bc = stats.inv_boxcox(pred_boxcox, lambda_val) + y_train.min() - 1
            r2_boxcox = r2_score(y_test, pred_original_bc)
        except:
            r2_boxcox = -1

        print(f"  对数变换 R²: {r2_log:.4f}")
        print(f"  Box-Cox变换 R²: {r2_boxcox:.4f}")

        if r2_log > self.best_r2 or r2_boxcox > self.best_r2:
            best_r2 = max(r2_log, r2_boxcox, self.best_r2)
            if r2_log == best_r2:
                return pred_original, r2_log, 'log'
            else:
                return pred_original_bc, r2_boxcox, 'boxcox'

        return None, self.best_r2, 'none'

    def outlier_robust_training(self, X_train, y_train, X_test, y_test):
        """异常值鲁棒训练"""
        print("尝试异常值鲁棒训练...")

        # 1. 识别异常值
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_train)

        # 2. 使用非异常值训练
        inlier_mask = outlier_labels == 1
        X_train_clean = X_train[inlier_mask]
        y_train_clean = y_train[inlier_mask]

        print(f"  移除异常值后训练集大小: {len(X_train_clean)}/{len(X_train)}")

        model_clean = ExtraTreesRegressor(n_estimators=1000, max_depth=35, random_state=42, n_jobs=-1)
        model_clean.fit(X_train_clean, y_train_clean)
        pred_clean = model_clean.predict(X_test)
        r2_clean = r2_score(y_test, pred_clean)

        print(f"  异常值鲁棒训练 R²: {r2_clean:.4f}")

        return pred_clean, r2_clean

    def neural_network_hybrid(self, X_train, y_train, X_test, y_test):
        """神经网络混合方法"""
        print("尝试神经网络混合方法...")

        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler

            # 标准化数据
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 神经网络模型
            nn_model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )

            nn_model.fit(X_train_scaled, y_train)
            nn_pred = nn_model.predict(X_test_scaled)
            nn_r2 = r2_score(y_test, nn_pred)

            print(f"  神经网络 R²: {nn_r2:.4f}")

            # 与ExtraTrees混合
            et_model = ExtraTreesRegressor(n_estimators=1000, max_depth=35, random_state=42, n_jobs=-1)
            et_model.fit(X_train, y_train)
            et_pred = et_model.predict(X_test)
            et_r2 = r2_score(y_test, et_pred)

            # 简单混合
            hybrid_pred = 0.7 * et_pred + 0.3 * nn_pred
            hybrid_r2 = r2_score(y_test, hybrid_pred)

            print(f"  混合模型 R²: {hybrid_r2:.4f}")

            return hybrid_pred, hybrid_r2

        except Exception as e:
            print(f"  神经网络方法失败: {e}")
            return None, self.best_r2


def explore_optimization_potential():
    """探索优化潜力"""
    print("=" * 70)
    print("探索ExtraTrees模型进一步优化潜力")
    print("=" * 70)

    # 加载数据
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_columns = [col for col in train_df.columns if col != 'value']
    X_train = train_df[feature_columns]
    y_train = train_df['value']
    X_test = test_df[feature_columns]
    y_test = test_df['value']

    # 分析潜力
    analyzer = AdvancedOptimizationStrategies()
    high_corr_count = analyzer.analyze_remaining_potential(X_train, y_train, X_test, y_test)

    print(f"\n🎯 优化潜力评估:")
    if high_corr_count < 10:
        print("  ✅ 特征与目标相关性较低，特征工程有较大潜力")
    else:
        print("  ✅ 存在多个相关特征，模型复杂度可能成为瓶颈")

    print("  ✅ 可以尝试更高级的集成架构")
    print("  ✅ 目标变量变换可能改善模型性能")
    print("  ✅ 异常值处理可能提升模型鲁棒性")
    print("  ✅ 神经网络混合方法值得尝试")


if __name__ == "__main__":
    explore_optimization_potential()