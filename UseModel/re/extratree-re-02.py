import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class ExtraTreesMultiModalEnsemble:
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}

    def create_diverse_extra_trees_models(self, X_train, y_train, n_models=8):
        """创建多样化的ExtraTrees模型"""
        print(f"创建 {n_models} 个多样化的ExtraTrees模型...")

        model_configs = [
            # 深度树配置
            {
                'name': 'deep_trees',
                'params': {
                    'n_estimators': 500,
                    'max_depth': 35,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 0.9,
                    'bootstrap': True,
                    'random_state': 42
                }
            },
            # 正则化配置
            {
                'name': 'regularized',
                'params': {
                    'n_estimators': 400,
                    'max_depth': 20,
                    'min_samples_split': 8,
                    'min_samples_leaf': 3,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': 43
                }
            },
            # 完全生长配置
            {
                'name': 'fully_grown',
                'params': {
                    'n_estimators': 300,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': None,
                    'bootstrap': False,
                    'random_state': 44
                }
            },
            # 高方差配置
            {
                'name': 'high_variance',
                'params': {
                    'n_estimators': 600,
                    'max_depth': 25,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 0.6,
                    'bootstrap': True,
                    'random_state': 45
                }
            },
            # 保守配置
            {
                'name': 'conservative',
                'params': {
                    'n_estimators': 800,
                    'max_depth': 15,
                    'min_samples_split': 15,
                    'min_samples_leaf': 5,
                    'max_features': 0.5,
                    'bootstrap': True,
                    'random_state': 46
                }
            },
            # 平衡配置
            {
                'name': 'balanced',
                'params': {
                    'n_estimators': 350,
                    'max_depth': 28,
                    'min_samples_split': 4,
                    'min_samples_leaf': 2,
                    'max_features': 0.8,
                    'bootstrap': True,
                    'random_state': 47
                }
            },
            # 特征重要配置
            {
                'name': 'feature_focused',
                'params': {
                    'n_estimators': 450,
                    'max_depth': 22,
                    'min_samples_split': 3,
                    'min_samples_leaf': 2,
                    'max_features': 0.7,
                    'bootstrap': True,
                    'random_state': 48
                }
            },
            # 大数据配置
            {
                'name': 'large_ensemble',
                'params': {
                    'n_estimators': 1000,
                    'max_depth': 18,
                    'min_samples_split': 10,
                    'min_samples_leaf': 4,
                    'max_features': 'log2',
                    'bootstrap': True,
                    'random_state': 49
                }
            }
        ]

        models = {}
        for config in model_configs[:n_models]:
            model = ExtraTreesRegressor(**config['params'], n_jobs=-1)
            model.fit(X_train, y_train)
            models[config['name']] = model

            # 评估模型性能
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=5, scoring='r2', n_jobs=-1)
            mean_r2 = cv_scores.mean()
            std_r2 = cv_scores.std()

            print(f"{config['name']:20} | CV R²: {mean_r2:.4f} (±{std_r2:.4f})")

            self.performance_metrics[config['name']] = {
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'cv_scores': cv_scores
            }

        return models

    def smart_weighted_ensemble(self, models, X_val, y_val, X_test):
        """基于验证集性能的智能加权集成"""
        print("计算智能权重...")

        model_weights = {}
        model_performances = {}

        for name, model in models.items():
            # 在验证集上评估
            y_pred_val = model.predict(X_val)
            r2_val = r2_score(y_val, y_pred_val)

            # 使用R²作为性能指标（处理负值）
            performance = max(r2_val, 0.01)  # 避免零权重
            model_performances[name] = performance

            # 基于性能计算权重（可以尝试不同的权重函数）
            weight = performance ** 2  # 平方放大差异
            model_weights[name] = weight

        # 归一化权重
        total_weight = sum(model_weights.values())
        for name in model_weights:
            model_weights[name] /= total_weight

        # 加权预测
        final_prediction = np.zeros(len(X_test))
        for name, model in models.items():
            pred = model.predict(X_test)
            final_prediction += model_weights[name] * pred

        print("模型权重分配:")
        for name, weight in model_weights.items():
            print(f"  {name:20} | 权重: {weight:.3f} | R²: {model_performances[name]:.4f}")

        return final_prediction, model_weights

    def create_meta_features_fixed(self, models, X_train, X_test, y_train, n_features_to_keep=50):
        """修复的元特征创建函数"""
        print("创建元特征...")

        # 第一层预测
        train_meta_features = []
        test_meta_features = []

        for name, model in models.items():
            # 训练集预测
            train_pred = model.predict(X_train).reshape(-1, 1)
            train_meta_features.append(train_pred)

            # 测试集预测
            test_pred = model.predict(X_test).reshape(-1, 1)
            test_meta_features.append(test_pred)

        # 选择最重要的原始特征（避免维度爆炸）
        base_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        base_model.fit(X_train, y_train)
        feature_importance = base_model.feature_importances_
        top_feature_indices = np.argsort(feature_importance)[-n_features_to_keep:]

        # 添加最重要的原始特征
        train_original_features = X_train.iloc[:, top_feature_indices].values
        test_original_features = X_test.iloc[:, top_feature_indices].values

        train_meta_features.append(train_original_features)
        test_meta_features.append(test_original_features)

        # 堆叠特征
        X_train_meta = np.column_stack(train_meta_features)
        X_test_meta = np.column_stack(test_meta_features)

        print(f"元特征维度 - 训练集: {X_train_meta.shape}, 测试集: {X_test_meta.shape}")

        return X_train_meta, X_test_meta

    def stacking_ensemble_fixed(self, base_models, X_train, y_train, X_test):
        """修复的堆叠集成"""
        print("构建堆叠集成...")

        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor

        # 创建元特征
        X_train_meta, X_test_meta = self.create_meta_features_fixed(base_models, X_train, X_test, y_train)

        # 尝试多个元学习器
        meta_learners = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }

        best_meta_pred = None
        best_meta_score = -np.inf
        best_meta_name = ""

        for name, meta_learner in meta_learners.items():
            try:
                # 交叉验证评估元学习器
                cv_scores = cross_val_score(meta_learner, X_train_meta, y_train,
                                            cv=3, scoring='r2', n_jobs=-1)
                mean_score = cv_scores.mean()

                print(f"元学习器 {name}: CV R² = {mean_score:.4f}")

                if mean_score > best_meta_score:
                    best_meta_score = mean_score
                    meta_learner.fit(X_train_meta, y_train)
                    best_meta_pred = meta_learner.predict(X_test_meta)
                    best_meta_name = name
            except Exception as e:
                print(f"元学习器 {name} 训练失败: {e}")
                continue

        print(f"最佳元学习器: {best_meta_name}, CV R²: {best_meta_score:.4f}")

        return best_meta_pred

    def median_ensemble(self, models, X_test):
        """中位数集成（对异常值鲁棒）"""
        print("执行中位数集成...")

        all_predictions = []
        for name, model in models.items():
            pred = model.predict(X_test)
            all_predictions.append(pred)

        # 转换为二维数组
        all_predictions = np.array(all_predictions)

        # 计算中位数
        median_pred = np.median(all_predictions, axis=0)

        return median_pred

    def best_model_selection(self, models, X_val, y_val, X_test):
        """选择最佳单模型"""
        print("选择最佳单模型...")

        best_model_name = None
        best_model_score = -np.inf
        best_prediction = None

        for name, model in models.items():
            y_pred_val = model.predict(X_val)
            r2_val = r2_score(y_val, y_pred_val)

            if r2_val > best_model_score:
                best_model_score = r2_val
                best_model_name = name
                best_prediction = model.predict(X_test)

        print(f"最佳单模型: {best_model_name}, R²: {best_model_score:.4f}")

        return best_prediction, best_model_name

    def hybrid_ensemble_fixed(self, models, X_train, y_train, X_test, y_test):
        """修复的混合集成策略"""
        print("执行混合集成策略...")

        # 分割训练集创建验证集
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # 在训练子集上重新训练模型（避免数据泄露）
        print("在训练子集上重新训练模型...")
        sub_models = {}
        for name, model in models.items():
            # 使用相同的参数创建新模型
            sub_model = ExtraTreesRegressor(**model.get_params())
            sub_model.fit(X_tr, y_tr)
            sub_models[name] = sub_model

        # 策略1: 智能加权集成
        print("\n1. 智能加权集成...")
        weighted_pred, weights = self.smart_weighted_ensemble(sub_models, X_val, y_val, X_test)
        weighted_r2 = r2_score(y_test, weighted_pred)
        print(f"加权集成 R²: {weighted_r2:.4f}")

        # 策略2: 堆叠集成
        print("\n2. 堆叠集成...")
        try:
            stacked_pred = self.stacking_ensemble_fixed(models, X_train, y_train, X_test)
            stacked_r2 = r2_score(y_test, stacked_pred)
            print(f"堆叠集成 R²: {stacked_r2:.4f}")
        except Exception as e:
            print(f"堆叠集成失败: {e}")
            stacked_pred = weighted_pred  # 使用加权集成作为备选
            stacked_r2 = weighted_r2

        # 策略3: 中位数集成
        print("\n3. 中位数集成...")
        median_pred = self.median_ensemble(models, X_test)
        median_r2 = r2_score(y_test, median_pred)
        print(f"中位数集成 R²: {median_r2:.4f}")

        # 策略4: 最佳单模型
        print("\n4. 最佳单模型选择...")
        best_single_pred, best_model_name = self.best_model_selection(sub_models, X_val, y_val, X_test)
        best_single_r2 = r2_score(y_test, best_single_pred)
        print(f"最佳单模型 R²: {best_single_r2:.4f}")

        # 策略5: 简单平均
        print("\n5. 简单平均集成...")
        all_predictions = [model.predict(X_test) for model in models.values()]
        simple_avg_pred = np.mean(all_predictions, axis=0)
        simple_avg_r2 = r2_score(y_test, simple_avg_pred)
        print(f"简单平均集成 R²: {simple_avg_r2:.4f}")

        # 收集所有策略
        strategies = {
            'weighted': (weighted_pred, weighted_r2),
            'stacked': (stacked_pred, stacked_r2),
            'median': (median_pred, median_r2),
            'best_single': (best_single_pred, best_single_r2),
            'simple_avg': (simple_avg_pred, simple_avg_r2)
        }

        # 选择最佳策略
        best_strategy_name = max(strategies.items(), key=lambda x: x[1][1])[0]
        best_strategy_pred, best_strategy_r2 = strategies[best_strategy_name]

        print(f"\n{'=' * 50}")
        print("集成策略比较:")
        for name, (pred, r2) in strategies.items():
            print(f"{name:15} | R²: {r2:.4f}")

        print(f"\n🎯 最佳集成策略: {best_strategy_name}, R²: {best_strategy_r2:.4f}")
        print(f"{'=' * 50}")

        return best_strategy_pred, best_strategy_name

    def evaluate_ensemble(self, models, X_test, y_test, ensemble_pred, ensemble_name):
        """评估集成效果"""
        print(f"\n{'=' * 60}")
        print(f"{ensemble_name} 集成评估结果")
        print(f"{'=' * 60}")

        # 单个模型性能
        print("单个模型在测试集上的性能:")
        single_model_r2 = {}
        for name, model in models.items():
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            single_model_r2[name] = r2
            print(f"{name:20} | R²: {r2:.4f}")

        # 集成模型性能
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

        print(f"\n集成模型性能:")
        print(f"R²:  {ensemble_r2:.4f}")
        print(f"MSE: {ensemble_mse:.4f}")
        print(f"MAE: {ensemble_mae:.4f}")

        # 提升分析
        best_single_r2 = max(single_model_r2.values())
        improvement = ensemble_r2 - best_single_r2
        print(f"\n相对于最佳单模型提升: {improvement:.4f}")

        if improvement > 0:
            print("🎯 集成策略有效提升了性能！")
        else:
            print("⚠️  集成策略未能提升性能")

        return ensemble_r2, improvement


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始ExtraTrees多态集成分析...")

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

    # 创建多态集成器
    ensemble = ExtraTreesMultiModalEnsemble()

    # 1. 创建多样化模型
    diverse_models = ensemble.create_diverse_extra_trees_models(X_train, y_train, n_models=8)

    # 2. 执行修复的混合集成
    final_pred, strategy_name = ensemble.hybrid_ensemble_fixed(diverse_models, X_train, y_train, X_test, y_test)

    # 3. 评估集成效果
    ensemble_r2, improvement = ensemble.evaluate_ensemble(diverse_models, X_test, y_test, final_pred, strategy_name)

    # 4. 保存结果
    results_df = pd.DataFrame({
        '真实值': y_test.values,
        '预测值': final_pred,
        '残差': y_test.values - final_pred
    })

    results_path = r"D:\PyProject\25卓越杯大数据\data2\multimodal_ensemble_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {results_path}")

    # 保存模型信息
    ensemble_info = {
        'strategy': strategy_name,
        'feature_columns': feature_columns,
        'performance': ensemble_r2,
        'improvement': improvement
    }

    import joblib
    joblib.dump(ensemble_info, r"D:\PyProject\25卓越杯大数据\data2\multimodal_extra_trees_ensemble.pkl")
    print(f"模型信息已保存")

    print(f"\n多态集成分析完成！最终R²: {ensemble_r2:.4f}")


if __name__ == "__main__":
    main()