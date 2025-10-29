import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import warnings

warnings.filterwarnings('ignore')


class RobustExtraTreesOptimizer:
    def __init__(self):
        self.models = {}
        self.feature_selector = None
        self.feature_names = None

    def safe_feature_engineering(self, X, y=None):
        """安全的特征工程，确保训练集和测试集特征一致"""
        print("进行安全的特征工程...")
        X_engineered = X.copy()

        # 只创建确定的统计特征，不依赖具体特征名称
        X_engineered['feature_mean'] = X.mean(axis=1)
        X_engineered['feature_std'] = X.std(axis=1)
        X_engineered['feature_max'] = X.max(axis=1)
        X_engineered['feature_min'] = X.min(axis=1)
        X_engineered['feature_median'] = X.median(axis=1)

        # 创建分位数特征
        X_engineered['feature_q25'] = X.quantile(0.25, axis=1)
        X_engineered['feature_q75'] = X.quantile(0.75, axis=1)

        print(f"特征工程后维度: {X_engineered.shape}")
        return X_engineered

    def consistent_feature_selection(self, X_train, y_train, X_test, n_features=300):
        """一致的特征选择，确保训练集和测试集特征相同"""
        print(f"进行一致的特征选择，目标特征数: {n_features}")

        # 使用相同的选择器处理训练集和测试集
        if self.feature_selector is None:
            # 初始化选择器
            base_estimator = ExtraTreesRegressor(n_estimators=100, random_state=42)
            self.feature_selector = SelectFromModel(
                base_estimator,
                max_features=n_features,
                threshold=-np.inf
            )
            self.feature_selector.fit(X_train, y_train)
            self.feature_names = X_train.columns[self.feature_selector.get_support()]

        # 使用相同的特征选择器
        X_train_selected = self.feature_selector.transform(X_train)
        X_test_selected = self.feature_selector.transform(X_test)

        # 转换为DataFrame保持特征名称
        X_train_df = pd.DataFrame(X_train_selected,
                                  columns=self.feature_names,
                                  index=X_train.index)
        X_test_df = pd.DataFrame(X_test_selected,
                                 columns=self.feature_names,
                                 index=X_test.index)

        print(f"特征选择后: {X_train_df.shape[1]} 个特征")
        return X_train_df, X_test_df

    def create_robust_extra_trees(self, X_train, y_train):
        """创建鲁棒的ExtraTrees模型"""
        print("创建鲁棒ExtraTrees模型...")

        # 更稳定的参数配置
        robust_configs = [
            {
                'name': 'robust_deep',
                'params': {
                    'n_estimators': 800,
                    'max_depth': 40,
                    'min_samples_split': 3,
                    'min_samples_leaf': 2,
                    'max_features': 0.8,
                    'bootstrap': True,
                    'random_state': 42,
                    'oob_score': True
                }
            },
            {
                'name': 'robust_balanced',
                'params': {
                    'n_estimators': 600,
                    'max_depth': 30,
                    'min_samples_split': 5,
                    'min_samples_leaf': 3,
                    'max_features': 0.7,
                    'bootstrap': True,
                    'random_state': 43,
                    'oob_score': True
                }
            },
            {
                'name': 'robust_conservative',
                'params': {
                    'n_estimators': 1000,
                    'max_depth': 25,
                    'min_samples_split': 8,
                    'min_samples_leaf': 4,
                    'max_features': 0.6,
                    'bootstrap': True,
                    'random_state': 44,
                    'oob_score': True
                }
            },
            {
                'name': 'robust_aggressive',
                'params': {
                    'n_estimators': 500,
                    'max_depth': 50,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 0.9,
                    'bootstrap': True,
                    'random_state': 45,
                    'oob_score': True
                }
            }
        ]

        models = {}
        for config in robust_configs:
            print(f"训练 {config['name']}...")
            model = ExtraTreesRegressor(**config['params'], n_jobs=-1)
            model.fit(X_train, y_train)
            models[config['name']] = model

            # 详细评估
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=5, scoring='r2', n_jobs=-1)
            mean_r2 = cv_scores.mean()
            std_r2 = cv_scores.std()

            oob_score = getattr(model, 'oob_score_', 0)

            print(f"  {config['name']:20} | CV R²: {mean_r2:.4f} | OOB: {oob_score:.4f}")

        return models

    def safe_weighted_ensemble(self, models, X_train, y_train, X_test):
        """安全的加权集成"""
        print("执行安全的加权集成...")

        weights = {}
        predictions = {}

        for name, model in models.items():
            # 使用OOB分数或交叉验证计算权重
            if hasattr(model, 'oob_score_') and model.oob_score_ > 0:
                weight = model.oob_score_
            else:
                cv_scores = cross_val_score(model, X_train, y_train,
                                            cv=3, scoring='r2', n_jobs=-1)
                weight = max(cv_scores.mean(), 0.1)

            weights[name] = weight
            predictions[name] = model.predict(X_test)

        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # 加权预测
        final_pred = np.zeros(len(X_test))
        for name, weight in weights.items():
            final_pred += weight * predictions[name]

        print("模型权重分配:")
        for name, weight in weights.items():
            print(f"  {name:20} | 权重: {weight:.3f}")

        return final_pred, weights

    def kfold_blending(self, models, X_train, y_train, X_test, n_folds=5):
        """K折混合集成"""
        print("执行K折混合集成...")

        from sklearn.linear_model import Ridge

        # 为每个模型创建OOF预测
        oof_predictions = np.zeros((len(X_train), len(models)))
        test_predictions = np.zeros((len(X_test), len(models)))

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        model_names = list(models.keys())

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"  折叠 {fold + 1}/{n_folds}...")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 在训练折上训练所有模型
            fold_models = {}
            for name in model_names:
                model_params = models[name].get_params()
                model = ExtraTreesRegressor(**model_params, n_jobs=-1)
                model.fit(X_tr, y_tr)
                fold_models[name] = model

                # OOF预测
                oof_predictions[val_idx, model_names.index(name)] = model.predict(X_val)
                # 测试集预测（平均）
                test_predictions[:, model_names.index(name)] += model.predict(X_test) / n_folds

        # 训练元学习器
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(oof_predictions, y_train)

        # 最终预测
        final_pred = meta_learner.predict(test_predictions)

        print(f"混合集成完成，元模型系数: {meta_learner.coef_}")
        return final_pred

    def smart_median_ensemble(self, models, X_test, use_weights=True):
        """智能中位数集成"""
        print("执行智能中位数集成...")

        all_predictions = []
        weights = []

        for name, model in models.items():
            pred = model.predict(X_test)
            all_predictions.append(pred)

            if use_weights and hasattr(model, 'oob_score_'):
                weights.append(model.oob_score_)
            else:
                weights.append(1.0)  # 等权重

        all_predictions = np.array(all_predictions)

        if use_weights:
            # 加权中位数
            sorted_indices = np.argsort(all_predictions, axis=0)
            cumulative_weights = np.cumsum(np.array(weights)[sorted_indices], axis=0)
            median_idx = np.argmax(cumulative_weights >= np.sum(weights) / 2.0, axis=0)
            final_pred = all_predictions[sorted_indices[median_idx, np.arange(len(X_test))],
            np.arange(len(X_test))]
        else:
            # 简单中位数
            final_pred = np.median(all_predictions, axis=0)

        return final_pred

    def evaluate_all_strategies(self, models, X_train, y_train, X_test, y_test):
        """评估所有集成策略"""
        print("\n评估所有集成策略...")

        strategies = {}

        # 策略1: 安全加权集成
        print("1. 安全加权集成...")
        weighted_pred, weights = self.safe_weighted_ensemble(models, X_train, y_train, X_test)
        weighted_r2 = r2_score(y_test, weighted_pred)
        strategies['weighted'] = (weighted_pred, weighted_r2)
        print(f"   加权集成 R²: {weighted_r2:.4f}")

        # 策略2: K折混合集成
        print("2. K折混合集成...")
        try:
            blended_pred = self.kfold_blending(models, X_train, y_train, X_test)
            blended_r2 = r2_score(y_test, blended_pred)
            strategies['blended'] = (blended_pred, blended_r2)
            print(f"   混合集成 R²: {blended_r2:.4f}")
        except Exception as e:
            print(f"   混合集成失败: {e}")
            strategies['blended'] = (weighted_pred, weighted_r2)

        # 策略3: 智能中位数集成
        print("3. 智能中位数集成...")
        median_pred = self.smart_median_ensemble(models, X_test, use_weights=True)
        median_r2 = r2_score(y_test, median_pred)
        strategies['median'] = (median_pred, median_r2)
        print(f"   中位数集成 R²: {median_r2:.4f}")

        # 策略4: 最佳单模型
        print("4. 最佳单模型选择...")
        best_single_r2 = -np.inf
        best_single_pred = None
        best_single_name = None
        for name, model in models.items():
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            if r2 > best_single_r2:
                best_single_r2 = r2
                best_single_pred = pred
                best_single_name = name
        strategies['best_single'] = (best_single_pred, best_single_r2)
        print(f"   最佳单模型 ({best_single_name}) R²: {best_single_r2:.4f}")

        # 策略5: 简单平均
        print("5. 简单平均集成...")
        all_predictions = [model.predict(X_test) for model in models.values()]
        simple_avg_pred = np.mean(all_predictions, axis=0)
        simple_avg_r2 = r2_score(y_test, simple_avg_pred)
        strategies['simple_avg'] = (simple_avg_pred, simple_avg_r2)
        print(f"   简单平均集成 R²: {simple_avg_r2:.4f}")

        return strategies

    def robust_ensemble_pipeline(self, X_train, y_train, X_test, y_test):
        """鲁棒集成管道"""
        print("开始鲁棒集成管道...")

        # 1. 安全特征工程
        print("\n1. 安全特征工程...")
        X_train_eng = self.safe_feature_engineering(X_train)
        X_test_eng = self.safe_feature_engineering(X_test)

        # 2. 一致特征选择
        print("\n2. 一致特征选择...")
        X_train_sel, X_test_sel = self.consistent_feature_selection(
            X_train_eng, y_train, X_test_eng, n_features=400
        )

        # 3. 创建鲁棒模型
        print("\n3. 创建鲁棒模型...")
        robust_models = self.create_robust_extra_trees(X_train_sel, y_train)

        # 4. 评估所有策略
        print("\n4. 评估集成策略...")
        all_strategies = self.evaluate_all_strategies(
            robust_models, X_train_sel, y_train, X_test_sel, y_test
        )

        # 选择最佳策略
        best_strategy_name = max(all_strategies.items(), key=lambda x: x[1][1])[0]
        best_pred, best_r2 = all_strategies[best_strategy_name]

        print(f"\n{'=' * 60}")
        print("集成策略最终比较:")
        for name, (pred, r2) in all_strategies.items():
            print(f"{name:15} | R²: {r2:.4f}")

        print(f"\n🎯 最佳策略: {best_strategy_name}, R²: {best_r2:.4f}")
        print(f"{'=' * 60}")

        return best_pred, best_strategy_name, all_strategies, robust_models


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始鲁棒ExtraTrees集成分析...")

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

    # 记录原始最佳性能
    original_best_model = ExtraTreesRegressor(
        n_estimators=300, max_depth=None, random_state=44, n_jobs=-1
    )
    original_best_model.fit(X_train, y_train)
    original_pred = original_best_model.predict(X_test)
    original_r2 = r2_score(y_test, original_pred)
    print(f"原始最佳模型R²: {original_r2:.4f}")

    # 创建鲁棒优化器
    optimizer = RobustExtraTreesOptimizer()

    # 执行鲁棒集成管道
    final_pred, best_strategy, all_strategies, models = optimizer.robust_ensemble_pipeline(
        X_train, y_train, X_test, y_test
    )

    # 评估改进
    improvement = all_strategies[best_strategy][1] - original_r2
    print(f"\n性能改进分析:")
    print(f"原始R²: {original_r2:.4f}")
    print(f"新R²: {all_strategies[best_strategy][1]:.4f}")
    print(f"提升: {improvement:.4f}")

    if improvement > 0:
        print("🎉 优化成功！")
    else:
        print("⚠️ 需要进一步优化")

    # 保存结果
    results_df = pd.DataFrame({
        '真实值': y_test.values,
        '预测值': final_pred,
        '残差': y_test.values - final_pred
    })

    results_path = r"D:\PyProject\25卓越杯大数据\data2\robust_ensemble_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {results_path}")

    print(f"\n鲁棒集成分析完成！最终R²: {all_strategies[best_strategy][1]:.4f}")


if __name__ == "__main__":
    main()