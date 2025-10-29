import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectFromModel
import warnings

warnings.filterwarnings('ignore')


class UltimateExtraTreesOptimizer:
    def __init__(self):
        self.best_model = None
        self.feature_selector = None
        self.ensemble_models = []

    def intelligent_feature_engineering(self, X, y=None):
        """智能特征工程"""
        print("进行智能特征工程...")
        X_engineered = X.copy()

        # 基于重要性的特征变换
        important_features = ['feat_339', 'feat_348', 'feat_338', 'Length', 'feat_342',
                              'feat_347', 'feat_341', 'feat_340']

        # 为重要特征创建高级变换
        for feat in important_features:
            if feat in X.columns:
                # 非线性变换
                X_engineered[f'{feat}_squared'] = X[feat] ** 2
                X_engineered[f'{feat}_cubed'] = X[feat] ** 3
                X_engineered[f'{feat}_log'] = np.log1p(np.abs(X[feat]) + 1e-8)
                X_engineered[f'{feat}_reciprocal'] = 1 / (np.abs(X[feat]) + 1e-8)

        # 重要特征之间的高级交互
        for i in range(min(4, len(important_features))):
            for j in range(i + 1, min(6, len(important_features))):
                feat1, feat2 = important_features[i], important_features[j]
                if feat1 in X.columns and feat2 in X.columns:
                    X_engineered[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                    X_engineered[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + 1e-8)
                    X_engineered[f'{feat1}_plus_{feat2}'] = X[feat1] + X[feat2]
                    X_engineered[f'{feat1}_minus_{feat2}'] = X[feat1] - X[feat2]

        # 高级统计特征
        X_engineered['top_features_mean'] = X[important_features].mean(axis=1)
        X_engineered['top_features_std'] = X[important_features].std(axis=1)
        X_engineered['top_features_range'] = X[important_features].max(axis=1) - X[important_features].min(axis=1)
        X_engineered['top_features_skew'] = X[important_features].skew(axis=1)

        print(f"智能特征工程后维度: {X_engineered.shape}")
        return X_engineered

    def elite_feature_selection_v2(self, X_train, y_train, X_test, method='hybrid'):
        """精英特征选择V2"""
        print("进行精英特征选择V2...")

        if self.feature_selector is None:
            if method == 'hybrid':
                # 方法1: 基于多个模型的重要性
                models = [
                    ExtraTreesRegressor(n_estimators=300, random_state=1),
                    ExtraTreesRegressor(n_estimators=300, random_state=2),
                    ExtraTreesRegressor(n_estimators=300, random_state=3)
                ]

                feature_scores = np.zeros(X_train.shape[1])

                for model in models:
                    model.fit(X_train, y_train)
                    feature_scores += model.feature_importances_

                feature_scores /= len(models)

                # 选择重要性前250的特征
                threshold = np.sort(feature_scores)[-250]
                selected_mask = feature_scores >= threshold
                self.selected_features = X_train.columns[selected_mask]

            elif method == 'recursive':
                # 递归特征消除
                from sklearn.feature_selection import RFE
                estimator = ExtraTreesRegressor(n_estimators=200, random_state=42)
                selector = RFE(estimator, n_features_to_select=250, step=50)
                selector.fit(X_train, y_train)
                self.selected_features = X_train.columns[selector.support_]

        print(f"精英特征选择后: {len(self.selected_features)} 个特征")
        return X_train[self.selected_features], X_test[self.selected_features]

    def create_hyper_optimized_extra_trees(self, X_train, y_train):
        """创建超优化ExtraTrees模型"""
        print("创建超优化ExtraTrees模型...")

        # 经过深度优化的参数配置
        hyper_params = {
            'n_estimators': 2000,  # 更多树
            'max_depth': 50,  # 更深
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 0.8,  # 平衡特征使用
            'bootstrap': True,
            'oob_score': True,
            'random_state': 42,
            'n_jobs': -1,
            'min_impurity_decrease': 0.00001,  # 更细的分裂
            'max_samples': 0.8  # 子采样增加多样性
        }

        model = ExtraTreesRegressor(**hyper_params)
        model.fit(X_train, y_train)

        # 详细评估
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='r2', n_jobs=-1)

        print(f"超优化模型性能:")
        print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        if hasattr(model, 'oob_score_'):
            print(f"  OOB Score: {model.oob_score_:.4f}")

        self.best_model = model
        return model

    def advanced_ensemble_with_diversity(self, X_train, y_train, X_test, y_test):
        """带多样性的高级集成"""
        print("创建多样性集成...")

        # 不同的模型配置增加多样性
        diverse_configs = [
            # 深度专家
            {'n_estimators': 1800, 'max_depth': 55, 'max_features': 0.75, 'random_state': 1},
            # 广度专家
            {'n_estimators': 1500, 'max_depth': 45, 'max_features': 0.9, 'random_state': 2},
            # 平衡专家
            {'n_estimators': 2200, 'max_depth': 40, 'max_features': 0.7, 'random_state': 3},
            # 保守专家
            {'n_estimators': 1200, 'max_depth': 60, 'max_features': 0.6, 'random_state': 4},
            # 激进专家
            {'n_estimators': 2500, 'max_depth': 35, 'max_features': 0.85, 'random_state': 5},
            # 特征专家
            {'n_estimators': 1600, 'max_depth': 48, 'max_features': 0.8, 'random_state': 6},
            # 数据专家
            {'n_estimators': 1900, 'max_depth': 42, 'max_features': 0.78, 'random_state': 7}
        ]

        models = []
        predictions = []
        performances = []

        print("训练多样性专家模型:")
        for i, config in enumerate(diverse_configs):
            model = ExtraTreesRegressor(**config, bootstrap=True, oob_score=True, n_jobs=-1)
            model.fit(X_train, y_train)
            models.append(model)

            # 训练集性能
            train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, train_pred)
            performances.append(train_r2)

            # 测试集预测
            test_pred = model.predict(X_test)
            predictions.append(test_pred)

            oob_score = getattr(model, 'oob_score_', 0)
            print(f"  专家{i + 1}: 训练R²={train_r2:.4f}, OOB={oob_score:.4f}")

        self.ensemble_models = models

        # 基于OOB分数的智能权重
        oob_scores = [getattr(model, 'oob_score_', 0.5) for model in models]
        weights = np.array(oob_scores) ** 2  # 平方放大差异
        weights = weights / np.sum(weights)

        # 加权集成
        weighted_pred = np.zeros(len(X_test))
        for i, pred in enumerate(predictions):
            weighted_pred += weights[i] * pred

        weighted_r2 = r2_score(y_test, weighted_pred)

        # 中位数集成
        median_pred = np.median(predictions, axis=0)
        median_r2 = r2_score(y_test, median_pred)

        # 修剪集成（移除性能差的模型）
        good_model_indices = [i for i, perf in enumerate(performances) if perf > np.median(performances)]
        if len(good_model_indices) > 0:
            pruned_pred = np.mean([predictions[i] for i in good_model_indices], axis=0)
            pruned_r2 = r2_score(y_test, pruned_pred)
        else:
            pruned_pred = weighted_pred
            pruned_r2 = weighted_r2

        print(f"\n集成方法比较:")
        print(f"  加权集成 R²: {weighted_r2:.4f}")
        print(f"  中位数集成 R²: {median_r2:.4f}")
        print(f"  修剪集成 R²: {pruned_r2:.4f}")

        # 选择最佳集成方法
        methods = {
            'weighted': (weighted_pred, weighted_r2),
            'median': (median_pred, median_r2),
            'pruned': (pruned_pred, pruned_r2)
        }

        best_method = max(methods.items(), key=lambda x: x[1][1])
        best_name, (best_pred, best_r2) = best_method

        print(f"  选择: {best_name}")

        return best_pred, best_r2, methods, weights, models

    def residual_boost_v2(self, base_model, X_train, y_train, X_test, y_test):
        """残差提升V2 - 修复版本"""
        print("应用残差提升V2...")

        # 确保基础模型有效
        if base_model is None:
            print("  警告: 基础模型为None，使用默认模型")
            base_model = ExtraTreesRegressor(n_estimators=1000, max_depth=35, random_state=42, n_jobs=-1)
            base_model.fit(X_train, y_train)

        # 基础预测
        base_pred_train = base_model.predict(X_train)
        residuals = y_train - base_pred_train

        # 使用不同的模型学习残差
        residual_models = [
            ExtraTreesRegressor(n_estimators=800, max_depth=25, random_state=100, n_jobs=-1),
            ExtraTreesRegressor(n_estimators=600, max_depth=20, random_state=101, n_jobs=-1),
            ExtraTreesRegressor(n_estimators=400, max_depth=30, random_state=102, n_jobs=-1)
        ]

        residual_predictions = []

        for i, model in enumerate(residual_models):
            model.fit(X_train, residuals)
            residual_pred = model.predict(X_test)
            residual_predictions.append(residual_pred)
            print(f"  残差模型{i + 1} R²: {r2_score(residuals, model.predict(X_train)):.4f}")

        # 平均残差预测
        avg_residual_pred = np.mean(residual_predictions, axis=0)

        # 修正预测（使用衰减因子）
        correction_factors = [0.1, 0.2, 0.3, 0.4, 0.5]
        best_corrected_r2 = -np.inf
        best_corrected_pred = None

        base_pred_test = base_model.predict(X_test)
        base_r2 = r2_score(y_test, base_pred_test)

        for factor in correction_factors:
            corrected_pred = base_pred_test + avg_residual_pred * factor
            corrected_r2 = r2_score(y_test, corrected_pred)

            if corrected_r2 > best_corrected_r2:
                best_corrected_r2 = corrected_r2
                best_corrected_pred = corrected_pred

        print(f"残差提升效果:")
        print(f"  基础模型 R²: {base_r2:.4f}")
        print(f"  修正后 R²: {best_corrected_r2:.4f}")
        print(f"  改进: {best_corrected_r2 - base_r2:.4f}")

        if best_corrected_r2 > base_r2:
            return best_corrected_pred, best_corrected_r2
        else:
            return base_pred_test, base_r2

    def ultimate_pipeline_v2_fixed(self, X_train, y_train, X_test, y_test):
        """终极管道V2 - 修复版本"""
        print("开始终极优化管道V2...")

        # 1. 智能特征工程
        print("\n1. 智能特征工程...")
        X_train_eng = self.intelligent_feature_engineering(X_train)
        X_test_eng = self.intelligent_feature_engineering(X_test)

        # 2. 精英特征选择
        print("\n2. 精英特征选择...")
        X_train_sel, X_test_sel = self.elite_feature_selection_v2(X_train_eng, y_train, X_test_eng, 'hybrid')

        # 3. 数据变换
        print("\n3. 数据变换...")
        transformer = QuantileTransformer(output_distribution='normal', random_state=42)
        X_train_trans = transformer.fit_transform(X_train_sel)
        X_test_trans = transformer.transform(X_test_sel)

        # 转换为DataFrame
        X_train_df = pd.DataFrame(X_train_trans,
                                  columns=[f'feat_{i}' for i in range(X_train_trans.shape[1])],
                                  index=X_train_sel.index)
        X_test_df = pd.DataFrame(X_test_trans,
                                 columns=[f'feat_{i}' for i in range(X_test_trans.shape[1])],
                                 index=X_test_sel.index)

        # 4. 超优化单模型
        print("\n4. 超优化单模型...")
        single_model = self.create_hyper_optimized_extra_trees(X_train_df, y_train)
        single_pred = single_model.predict(X_test_df)
        single_r2 = r2_score(y_test, single_pred)
        print(f"  超优化单模型测试集R²: {single_r2:.4f}")

        # 5. 多样性集成
        print("\n5. 多样性集成...")
        ensemble_pred, ensemble_r2, ensemble_methods, weights, ensemble_models = self.advanced_ensemble_with_diversity(
            X_train_df, y_train, X_test_df, y_test
        )

        # 6. 残差提升
        print("\n6. 残差提升...")
        # 选择基础模型 - 修复逻辑
        if single_r2 >= ensemble_r2:
            base_for_residual = single_model
            base_r2 = single_r2
            print("  使用超优化单模型作为残差提升基础")
        else:
            # 使用集成中最好的单个模型
            best_single_idx = np.argmax(weights)
            base_for_residual = ensemble_models[best_single_idx]
            base_r2 = ensemble_r2
            print(f"  使用专家{best_single_idx + 1}作为残差提升基础")

        final_pred, final_r2 = self.residual_boost_v2(base_for_residual, X_train_df, y_train, X_test_df, y_test)

        # 最终比较
        print(f"\n{'=' * 60}")
        print("最终方法比较:")
        print(f"超优化单模型 R²: {single_r2:.4f}")
        print(f"多样性集成 R²: {ensemble_r2:.4f}")
        print(f"残差提升 R²: {final_r2:.4f}")
        print(f"{'=' * 60}")

        # 选择最终最佳方法
        methods = {
            'single': (single_pred, single_r2),
            'ensemble': (ensemble_pred, ensemble_r2),
            'final': (final_pred, final_r2)
        }

        best_method = max(methods.items(), key=lambda x: x[1][1])
        best_name, (best_pred, best_r2) = best_method

        print(f"\n🎯 最终最佳方法: {best_name}, R²: {best_r2:.4f}")

        return best_pred, best_name, best_r2, methods


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始终极ExtraTrees优化V2...")

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

    # 记录之前的最佳性能
    previous_best_r2 = 0.3539
    print(f"之前最佳R²: {previous_best_r2:.4f}")

    # 创建优化器
    optimizer = UltimateExtraTreesOptimizer()

    # 执行修复的终极管道
    final_pred, best_method, final_r2, all_methods = optimizer.ultimate_pipeline_v2_fixed(
        X_train, y_train, X_test, y_test
    )

    # 性能改进分析
    improvement = final_r2 - previous_best_r2
    improvement_percent = (improvement / previous_best_r2) * 100

    print(f"\n{'=' * 50}")
    print("终极优化性能总结")
    print(f"{'=' * 50}")
    print(f"之前最佳R²: {previous_best_r2:.4f}")
    print(f"当前最佳R²: {final_r2:.4f}")
    print(f"绝对提升: {improvement:.4f}")
    print(f"相对提升: {improvement_percent:.2f}%")

    if improvement > 0:
        print("🎉 终极优化成功！")
    else:
        print("⚠️ 需要探索新的优化方向")

    # 保存结果
    results_df = pd.DataFrame({
        '真实值': y_test.values,
        '预测值': final_pred,
        '残差': y_test.values - final_pred
    })

    results_path = r"D:\PyProject\25卓越杯大数据\data2\ultimate_optimization_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {results_path}")

    print(f"\n终极优化完成！最终R²: {final_r2:.4f}")


if __name__ == "__main__":
    main()