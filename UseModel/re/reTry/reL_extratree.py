import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')


class ExtraTreesOptimizer:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    def comprehensive_parameter_optimization(self, X_train, y_train):
        """全面的参数优化"""
        print("进行ExtraTrees参数优化...")

        # 基础参数网格
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8, None],
            'bootstrap': [True, False]
        }

        # 使用随机搜索（更高效）
        base_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=50,  # 随机搜索50次
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        print(f"最优参数: {random_search.best_params_}")
        print(f"最优分数 (R²): {random_search.best_score_:.4f}")

        return random_search.best_estimator_

    def create_diverse_ensemble(self, X_train, y_train, n_models=5):
        """创建多样化的ExtraTrees集成"""
        print(f"创建包含 {n_models} 个模型的多样化集成...")

        # 不同的参数配置，增加模型多样性
        param_configs = [
            # 配置1: 深树，更多特征
            {
                'n_estimators': 300,
                'max_depth': 25,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 0.8,
                'bootstrap': True,
                'random_state': 42
            },
            # 配置2: 正则化更强的树
            {
                'n_estimators': 400,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 43
            },
            # 配置3: 完全生长的树
            {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None,
                'bootstrap': False,
                'random_state': 44
            },
            # 配置4: 保守配置
            {
                'n_estimators': 500,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'log2',
                'bootstrap': True,
                'random_state': 45
            },
            # 配置5: 平衡配置
            {
                'n_estimators': 350,
                'max_depth': 22,
                'min_samples_split': 3,
                'min_samples_leaf': 2,
                'max_features': 0.7,
                'bootstrap': True,
                'random_state': 46
            }
        ]

        models = {}
        for i, params in enumerate(param_configs[:n_models]):
            model = ExtraTreesRegressor(**params, n_jobs=-1)
            model.fit(X_train, y_train)
            models[f'et_model_{i + 1}'] = model

            # 存储特征重要性
            self.feature_importance[f'model_{i + 1}'] = model.feature_importances_

            # 交叉验证评估
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=5, scoring='r2', n_jobs=-1)
            print(f"模型 {i + 1} - 平均R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        self.models = models
        return models

    def calculate_optimal_weights(self, X_train, y_train, models):
        """基于性能计算最优权重"""
        print("计算模型最优权重...")

        model_performances = {}
        model_predictions = {}

        # 使用袋外分数或交叉验证评估每个模型
        for name, model in models.items():
            try:
                # 尝试获取袋外分数
                if hasattr(model, 'oob_score_'):
                    performance = model.oob_score_
                else:
                    # 使用交叉验证
                    cv_scores = cross_val_score(model, X_train, y_train,
                                                cv=5, scoring='r2', n_jobs=-1)
                    performance = cv_scores.mean()

                model_performances[name] = max(performance, 0.1)  # 避免负权重

            except:
                model_performances[name] = 0.1  # 默认权重

        # 基于性能计算权重（性能越好权重越高）
        total_performance = sum(model_performances.values())
        weights = {name: perf / total_performance for name, perf in model_performances.items()}

        print("模型权重分配:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")

        return weights

    def weighted_ensemble_predict(self, models, weights, X):
        """加权集成预测"""
        weighted_sum = None

        for name, model in models.items():
            pred = model.predict(X)
            if weighted_sum is None:
                weighted_sum = weights[name] * pred
            else:
                weighted_sum += weights[name] * pred

        return weighted_sum

    def adaptive_ensemble_predict(self, models, X, y_train, method='performance'):
        """自适应集成预测"""
        if method == 'performance':
            # 基于历史性能的固定权重
            weights = self.calculate_optimal_weights(X, y_train, models)
            return self.weighted_ensemble_predict(models, weights, X)

        elif method == 'dynamic':
            # 基于最近表现的动态权重（需要验证集）
            return self.dynamic_weighted_predict(models, X)

        elif method == 'median':
            # 中位数集成（对异常值更鲁棒）
            all_predictions = np.column_stack([model.predict(X) for model in models.values()])
            return np.median(all_predictions, axis=1)

    def bootstrap_aggregating(self, X_train, y_train, n_models=10):
        """Bootstrap Aggregating (Bagging)"""
        print(f"进行Bootstrap聚合，创建 {n_models} 个模型...")

        bagged_models = {}

        for i in range(n_models):
            # 自助采样
            X_boot, y_boot = resample(X_train, y_train, random_state=i)

            # 训练模型（使用不同的随机种子）
            model = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=42 + i,
                n_jobs=-1
            )
            model.fit(X_boot, y_boot)
            bagged_models[f'bag_model_{i + 1}'] = model

        return bagged_models

    def feature_subsampling_ensemble(self, X_train, y_train, feature_subsets=5):
        """特征子采样集成"""
        print(f"创建特征子采样集成，{feature_subsets} 个子集...")

        n_features = X_train.shape[1]
        subset_size = max(n_features // 2, n_features - 10)  # 使用一半特征

        feature_models = {}

        for i in range(feature_subsets):
            # 随机选择特征子集
            feature_indices = np.random.choice(n_features, subset_size, replace=False)
            X_subset = X_train.iloc[:, feature_indices]

            model = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=47 + i,
                n_jobs=-1
            )
            model.fit(X_subset, y_train)
            feature_models[f'feat_model_{i + 1}'] = (model, feature_indices)

        return feature_models

    def evaluate_ensemble_strategies(self, X_train, y_train, X_test, y_test):
        """评估不同的集成策略"""
        print("\n评估不同集成策略...")

        strategies_results = {}

        # 策略1: 参数优化单模型
        print("1. 参数优化单模型...")
        best_single_model = self.comprehensive_parameter_optimization(X_train, y_train)
        single_pred = best_single_model.predict(X_test)
        strategies_results['optimized_single'] = {
            'r2': r2_score(y_test, single_pred),
            'model': best_single_model
        }

        # 策略2: 多样化集成
        print("2. 多样化集成...")
        diverse_models = self.create_diverse_ensemble(X_train, y_train)
        diverse_weights = self.calculate_optimal_weights(X_train, y_train, diverse_models)
        diverse_pred = self.weighted_ensemble_predict(diverse_models, diverse_weights, X_test)
        strategies_results['diverse_ensemble'] = {
            'r2': r2_score(y_test, diverse_pred),
            'models': diverse_models,
            'weights': diverse_weights
        }

        # 策略3: Bagging集成
        print("3. Bagging集成...")
        bagged_models = self.bootstrap_aggregating(X_train, y_train)
        bagged_weights = self.calculate_optimal_weights(X_train, y_train, bagged_models)
        bagged_pred = self.weighted_ensemble_predict(bagged_models, bagged_weights, X_test)
        strategies_results['bagging_ensemble'] = {
            'r2': r2_score(y_test, bagged_pred),
            'models': bagged_models
        }

        # 策略4: 中位数集成
        print("4. 中位数集成...")
        median_pred = self.adaptive_ensemble_predict(diverse_models, X_test, y_train, 'median')
        strategies_results['median_ensemble'] = {
            'r2': r2_score(y_test, median_pred)
        }

        # 输出比较结果
        print("\n" + "=" * 60)
        print("集成策略性能比较")
        print("=" * 60)
        for strategy, result in strategies_results.items():
            print(f"{strategy:20} | R²: {result['r2']:.4f}")

        return strategies_results

    def get_feature_importance_consensus(self, models):
        """获取特征重要性的共识"""
        print("计算特征重要性共识...")

        all_importances = []
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                all_importances.append(model.feature_importances_)

        if all_importances:
            consensus_importance = np.mean(all_importances, axis=0)
            return consensus_importance
        else:
            return None


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始ExtraTrees参数优化和集成策略分析...")

    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 准备数据
    feature_columns = [col for col in train_df.columns if col != 'value']
    X_train = train_df[feature_columns]
    y_train = train_df['value']
    X_test = test_df[feature_columns]
    y_test = test_df['value']

    print(f"数据形状 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 创建优化器
    optimizer = ExtraTreesOptimizer()

    # 评估不同策略
    results = optimizer.evaluate_ensemble_strategies(X_train, y_train, X_test, y_test)

    # 选择最佳策略
    best_strategy = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\n最佳策略: {best_strategy[0]}, R²: {best_strategy[1]['r2']:.4f}")

    # 保存最佳模型
    if 'models' in best_strategy[1]:
        # 保存集成模型
        ensemble_info = {
            'models': best_strategy[1]['models'],
            'strategy': best_strategy[0],
            'feature_columns': feature_columns,
            'performance': best_strategy[1]['r2']
        }
        joblib.dump(ensemble_info, r"D:\PyProject\25卓越杯大数据\data2\optimized_extra_trees_ensemble.pkl")
    else:
        # 保存单模型
        joblib.dump(best_strategy[1]['model'], r"D:\PyProject\25卓越杯大数据\data2\optimized_extra_trees_model.pkl")

    print("优化完成！")


if __name__ == "__main__":
    main()