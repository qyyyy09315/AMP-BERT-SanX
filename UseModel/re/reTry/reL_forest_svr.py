import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RobustEnsembleModel:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.feature_selector = None
        self.best_ensemble = None

    def load_data(self, train_path, test_path, target_column='value'):
        """加载和预处理数据"""
        print("正在加载数据集...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")

        # 统一特征名
        train_df, test_df = self._unify_features(train_df, test_df, target_column)

        # 准备数据
        feature_columns = [col for col in train_df.columns if col != target_column]
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        return X_train, X_test, y_train, y_test, feature_columns

    def _unify_features(self, train_df, test_df, target_column):
        """统一训练集和测试集的特征"""
        train_features = set(train_df.columns) - {target_column}
        test_features = set(test_df.columns) - {target_column}

        common_features = list(train_features & test_features)

        if len(common_features) != len(train_features) or len(common_features) != len(test_features):
            print("警告：训练集和测试集特征不完全一致，使用共同特征")
            print(f"共同特征数量: {len(common_features)}")

        return train_df[common_features + [target_column]], test_df[common_features + [target_column]]

    def feature_engineering(self, X_train, X_test, y_train):
        """特征工程和选择"""
        print("\n进行特征选择...")

        # 使用随机森林进行特征重要性选择
        selector = RandomForestRegressor(n_estimators=100, random_state=42)
        selector.fit(X_train, y_train)

        # 选择重要性前80%的特征
        importances = selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        cumulative_importance = np.cumsum(importances[indices])
        n_features = np.argmax(cumulative_importance >= 0.8) + 1

        selected_features = X_train.columns[indices[:n_features]]
        print(f"从 {X_train.shape[1]} 个特征中选择 {n_features} 个重要特征")

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        return X_train_selected, X_test_selected, selected_features

    def optimize_models(self, X_train, y_train):
        """优化多个基础模型"""
        print("\n开始优化基础模型...")

        # 定义模型和参数网格
        models_config = {
            'lasso': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10],
                    'max_iter': [5000]
                }
            },
            'ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10]
                }
            },
            'svr': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', SVR())
                ]),
                'params': {
                    'svr__C': [0.1, 1, 10, 100],
                    'svr__epsilon': [0.01, 0.1, 0.5],
                    'svr__kernel': ['rbf']
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None]
                }
            }
        }

        best_models = {}

        for name, config in models_config.items():
            print(f"\n优化 {name}...")
            try:
                grid = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )

                grid.fit(X_train, y_train)
                best_models[name] = grid.best_estimator_
                best_score = -grid.best_score_

                print(f"{name} 最优参数: {grid.best_params_}")
                print(f"{name} 交叉验证MSE: {best_score:.4f}")

            except Exception as e:
                print(f"优化 {name} 时出错: {e}")
                # 使用默认参数
                best_models[name] = config['model']
                best_models[name].fit(X_train, y_train)

        return best_models

    def calculate_ensemble_weights(self, models, X_train, y_train, method='performance_based'):
        """计算集成权重"""
        print("\n计算模型权重...")

        if method == 'performance_based':
            return self._performance_based_weights(models, X_train, y_train)
        elif method == 'equal':
            return self._equal_weights(models)
        else:
            return self._stacking_weights(models, X_train, y_train)

    def _performance_based_weights(self, models, X_train, y_train):
        """基于性能的权重计算"""
        model_performances = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            try:
                # 使用R²分数作为性能指标
                scores = cross_val_score(model, X_train, y_train,
                                         cv=kf, scoring='r2', n_jobs=-1)
                mean_r2 = np.mean(scores)
                # 将R²转换为正权重（处理负R²情况）
                performance = max(0, mean_r2) + 0.1  # 加0.1避免除零
                model_performances[name] = performance
                print(f"{name} 平均R²: {mean_r2:.4f}")

            except Exception as e:
                print(f"评估 {name} 时出错: {e}")
                model_performances[name] = 0.1  # 默认权重

        # 计算权重
        total_performance = sum(model_performances.values())
        weights = {name: perf / total_performance for name, perf in model_performances.items()}

        print("\n模型权重分配:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")

        return weights

    def _equal_weights(self, models):
        """等权重分配"""
        n_models = len(models)
        return {name: 1.0 / n_models for name in models.keys()}

    def build_stacking_ensemble(self, models, X_train, y_train, X_test):
        """构建堆叠集成"""
        print("\n构建堆叠集成模型...")

        # 生成第一层预测
        train_predictions = []
        test_predictions = []
        model_names = []

        for name, model in models.items():
            print(f"训练 {name}...")
            try:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_predictions.append(train_pred)
                test_predictions.append(test_pred)
                model_names.append(name)

            except Exception as e:
                print(f"训练 {name} 时出错: {e}")
                continue

        if not train_predictions:
            raise ValueError("没有成功训练的模型")

        # 创建第二层特征
        X_train_stacked = np.column_stack(train_predictions)
        X_test_stacked = np.column_stack(test_predictions)

        # 使用线性回归作为元学习器
        from sklearn.linear_model import LinearRegression
        meta_model = LinearRegression()
        meta_model.fit(X_train_stacked, y_train)

        # 最终预测
        y_pred_stacked = meta_model.predict(X_test_stacked)

        print("堆叠集成模型构建完成")
        print(f"元模型系数: {meta_model.coef_}")

        return y_pred_stacked, meta_model

    def predict_ensemble(self, models, weights, X_test, method='weighted'):
        """集成预测"""
        if method == 'weighted':
            return self._weighted_prediction(models, weights, X_test)
        else:
            return self._median_prediction(models, X_test)

    def _weighted_prediction(self, models, weights, X_test):
        """加权预测"""
        weighted_sum = None

        for name, model in models.items():
            pred = model.predict(X_test)
            if weighted_sum is None:
                weighted_sum = weights[name] * pred
            else:
                weighted_sum += weights[name] * pred

        return weighted_sum

    def _median_prediction(self, models, X_test):
        """中位数预测（更鲁棒）"""
        all_predictions = []

        for name, model in models.items():
            pred = model.predict(X_test)
            all_predictions.append(pred)

        return np.median(all_predictions, axis=0)

    def evaluate_models(self, models, X_test, y_test, ensemble_pred):
        """全面评估模型性能"""
        print("\n" + "=" * 80)
        print("模型性能评估")
        print("=" * 80)

        results = {}

        # 评估单个模型
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                results[name] = {
                    'MSE': mse,
                    'R2': r2,
                    'MAE': mae
                }

                print(f"{name:15} | MSE: {mse:8.4f} | R²: {r2:7.4f} | MAE: {mae:7.4f}")

            except Exception as e:
                print(f"评估 {name} 时出错: {e}")

        # 评估集成模型
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

        results['ensemble'] = {
            'MSE': ensemble_mse,
            'R2': ensemble_r2,
            'MAE': ensemble_mae
        }

        print(f"{'集成模型':15} | MSE: {ensemble_mse:8.4f} | R²: {ensemble_r2:7.4f} | MAE: {ensemble_mae:7.4f}")
        print("=" * 80)

        return results

    def plot_comprehensive_results(self, models, X_test, y_test, ensemble_pred, results):
        """绘制综合结果图表"""
        print("\n生成可视化结果...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 所有模型预测对比
        model_names = list(models.keys()) + ['集成模型']
        all_predictions = [model.predict(X_test) for model in models.values()] + [ensemble_pred]

        for i, (name, pred) in enumerate(zip(model_names, all_predictions)):
            axes[0, 0].scatter(y_test, pred, alpha=0.6, label=name, s=30)
            r2 = r2_score(y_test, pred)
            axes[0, 0].text(0.05, 0.95 - i * 0.06, f'{name} R² = {r2:.3f}',
                            transform=axes[0, 0].transAxes, fontsize=8)

        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.8)
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title('所有模型预测对比')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 性能对比柱状图
        metrics = ['MSE', 'R2', 'MAE']
        x_pos = np.arange(len(model_names))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [results[name][metric] for name in model_names]
            axes[0, 1].bar(x_pos + i * width, values, width, label=metric, alpha=0.8)

        axes[0, 1].set_xlabel('模型')
        axes[0, 1].set_ylabel('数值')
        axes[0, 1].set_title('模型性能对比')
        axes[0, 1].set_xticks(x_pos + width)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 残差分析
        ensemble_residuals = y_test - ensemble_pred
        axes[0, 2].scatter(ensemble_pred, ensemble_residuals, alpha=0.6)
        axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('集成模型预测值')
        axes[0, 2].set_ylabel('残差')
        axes[0, 2].set_title('集成模型残差分析')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 预测值分布对比
        axes[1, 0].hist(y_test, bins=30, alpha=0.7, label='真实值', density=True)
        axes[1, 0].hist(ensemble_pred, bins=30, alpha=0.7, label='预测值', density=True)
        axes[1, 0].set_xlabel('值')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title('真实值与预测值分布对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 模型误差分布
        errors = []
        labels = []
        for name in models.keys():
            pred = models[name].predict(X_test)
            error = np.abs(y_test - pred)
            errors.extend(error)
            labels.extend([name] * len(error))

        error_df = pd.DataFrame({'误差': errors, '模型': labels})
        sns.boxplot(data=error_df, x='模型', y='误差', ax=axes[1, 1])
        axes[1, 1].set_title('各模型绝对误差分布')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # 6. 学习曲线（简化版）
        ensemble_r2 = results['ensemble']['R2']
        best_single_r2 = max([results[name]['R2'] for name in models.keys()])
        improvement = ensemble_r2 - best_single_r2

        models_for_plot = ['最佳单模型', '集成模型']
        r2_values = [best_single_r2, ensemble_r2]

        bars = axes[1, 2].bar(models_for_plot, r2_values, color=['lightblue', 'lightcoral'], alpha=0.8)
        axes[1, 2].set_ylabel('R² Score')
        axes[1, 2].set_title(f'集成提升: {improvement:.4f}')
        axes[1, 2].grid(True, alpha=0.3)

        # 在柱状图上添加数值
        for bar, value in zip(bars, r2_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始强健集成模型分析...")

    # 创建集成模型实例
    ensemble_model = RobustEnsembleModel()

    # 1. 加载数据
    X_train, X_test, y_train, y_test, feature_columns = ensemble_model.load_data(
        train_path, test_path
    )

    # 2. 特征工程（可选）
    # X_train, X_test, selected_features = ensemble_model.feature_engineering(
    #     X_train, X_test, y_train
    # )

    # 3. 优化多个模型
    best_models = ensemble_model.optimize_models(X_train, y_train)

    # 4. 计算权重并构建集成
    weights = ensemble_model.calculate_ensemble_weights(best_models, X_train, y_train)

    # 5. 集成预测
    ensemble_pred = ensemble_model.predict_ensemble(best_models, weights, X_test)

    # 6. 尝试堆叠集成
    try:
        stacked_pred, meta_model = ensemble_model.build_stacking_ensemble(
            best_models, X_train, y_train, X_test
        )
        # 使用堆叠集成结果
        ensemble_pred = stacked_pred
        print("使用堆叠集成结果")
    except Exception as e:
        print(f"堆叠集成失败，使用加权集成: {e}")

    # 7. 评估模型
    results = ensemble_model.evaluate_models(best_models, X_test, y_test, ensemble_pred)

    # 8. 可视化
    ensemble_model.plot_comprehensive_results(best_models, X_test, y_test, ensemble_pred, results)

    # 9. 保存结果
    output_dir = r"D:\PyProject\25卓越杯大数据\data2"

    # 保存模型
    model_info = {
        'models': best_models,
        'weights': weights,
        'feature_columns': feature_columns,
        'ensemble_pred': ensemble_pred
    }

    joblib.dump(model_info, f"{output_dir}/robust_ensemble_model.pkl")

    # 保存预测结果
    results_df = pd.DataFrame({
        '真实值': y_test,
        '集成预测值': ensemble_pred,
        '残差': y_test - ensemble_pred
    })

    # 添加各模型预测结果
    for name, model in best_models.items():
        results_df[f'{name}_预测'] = model.predict(X_test)

    results_df.to_csv(f"{output_dir}/robust_ensemble_results.csv", index=False, encoding='utf-8-sig')

    print(f"\n强健集成模型分析完成！")
    print(f"最佳单模型R²: {max([results[name]['R2'] for name in best_models.keys()]):.4f}")
    print(f"集成模型R²: {results['ensemble']['R2']:.4f}")


if __name__ == "__main__":
    main()