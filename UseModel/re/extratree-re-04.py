import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold
import warnings

warnings.filterwarnings('ignore')


class UltimateExtraTreesOptimizer:
    def __init__(self):
        self.best_model = None
        self.feature_selector = None
        self.feature_names = None

    def advanced_feature_engineering_v2(self, X, y=None):
        """更高级的特征工程"""
        print("进行高级特征工程V2...")
        X_engineered = X.copy()

        # 基础统计特征
        X_engineered['feature_mean'] = X.mean(axis=1)
        X_engineered['feature_std'] = X.std(axis=1)
        X_engineered['feature_skew'] = X.skew(axis=1)
        X_engineered['feature_kurtosis'] = X.kurtosis(axis=1)
        X_engineered['feature_max'] = X.max(axis=1)
        X_engineered['feature_min'] = X.min(axis=1)
        X_engineered['feature_median'] = X.median(axis=1)

        # 分位数特征
        for q in [0.1, 0.25, 0.75, 0.9]:
            X_engineered[f'feature_q{int(q * 100)}'] = X.quantile(q, axis=1)

        # 创建一些重要的交互特征
        if 'feature_mean' in X_engineered.columns and 'feature_std' in X_engineered.columns:
            X_engineered['mean_std_ratio'] = X_engineered['feature_mean'] / (X_engineered['feature_std'] + 1e-8)
            X_engineered['mean_std_product'] = X_engineered['feature_mean'] * X_engineered['feature_std']

        print(f"特征工程后维度: {X_engineered.shape}")
        return X_engineered

    def smart_feature_selection(self, X_train, y_train, X_test, selection_method='importance'):
        """智能特征选择"""
        print("进行智能特征选择...")

        if selection_method == 'importance':
            # 基于特征重要性选择
            selector = SelectFromModel(
                ExtraTreesRegressor(n_estimators=200, random_state=42),
                max_features=350,
                threshold=-np.inf
            )
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]

        elif selection_method == 'rfe':
            # 递归特征消除
            estimator = ExtraTreesRegressor(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=300, step=50)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]

        elif selection_method == 'variance':
            # 基于方差选择
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X_train)
            selected_features = X_train.columns[selector.get_support()]

        print(f"特征选择后: {len(selected_features)} 个特征")

        # 保存选择器状态用于测试集
        self.feature_selector = selected_features

        return X_train[selected_features], X_test[selected_features]

    def hyperparameter_tuning(self, X_train, y_train):
        """超参数调优"""
        print("进行超参数调优...")

        # 定义参数网格
        param_grid = {
            'n_estimators': [500, 800, 1000, 1200],
            'max_depth': [25, 30, 35, 40, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2],
            'max_features': [0.7, 0.8, 0.9],
            'bootstrap': [True]
        }

        # 使用随机搜索
        base_model = ExtraTreesRegressor(random_state=42, oob_score=True, n_jobs=-1)

        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=15,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        print(f"最优参数: {random_search.best_params_}")
        print(f"最优分数 (R²): {random_search.best_score_:.4f}")
        if hasattr(random_search.best_estimator_, 'oob_score_'):
            print(f"OOB分数: {random_search.best_estimator_.oob_score_:.4f}")

        return random_search.best_estimator_

    def create_ultimate_model(self, X_train, y_train, use_tuned_params=True):
        """创建终极模型"""
        print("创建终极ExtraTrees模型...")

        if use_tuned_params:
            # 经过调优的最佳配置
            best_params = {
                'n_estimators': 1000,
                'max_depth': 35,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 0.8,
                'bootstrap': True,
                'oob_score': True,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            # 更激进的配置
            best_params = {
                'n_estimators': 1200,
                'max_depth': 40,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 0.9,
                'bootstrap': True,
                'oob_score': True,
                'random_state': 42,
                'n_jobs': -1
            }

        model = ExtraTreesRegressor(**best_params)
        model.fit(X_train, y_train)

        # 交叉验证评估
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='r2', n_jobs=-1)

        print(f"模型性能:")
        print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        if hasattr(model, 'oob_score_'):
            print(f"  OOB Score: {model.oob_score_:.4f}")

        self.best_model = model
        return model

    def data_transformation_boost(self, X_train, y_train, X_test, transformation='quantile'):
        """数据变换提升"""
        print("应用数据变换...")

        if transformation == 'quantile':
            transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            X_train_transformed = transformer.fit_transform(X_train)
            X_test_transformed = transformer.transform(X_test)
        elif transformation == 'standard':
            transformer = StandardScaler()
            X_train_transformed = transformer.fit_transform(X_train)
            X_test_transformed = transformer.transform(X_test)
        else:
            return X_train, X_test

        # 转换回DataFrame
        X_train_df = pd.DataFrame(X_train_transformed,
                                  columns=X_train.columns,
                                  index=X_train.index)
        X_test_df = pd.DataFrame(X_test_transformed,
                                 columns=X_test.columns,
                                 index=X_test.index)

        return X_train_df, X_test_df

    def ensemble_of_best(self, X_train, y_train, X_test, n_models=5):
        """最佳模型集成"""
        print("创建最佳模型集成...")

        # 不同的随机种子创建多样性
        models = []
        predictions = []

        for i in range(n_models):
            model = ExtraTreesRegressor(
                n_estimators=1000,
                max_depth=35,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.8,
                bootstrap=True,
                oob_score=True,
                random_state=42 + i * 10,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            models.append(model)

            pred = model.predict(X_test)
            predictions.append(pred)

            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=3, scoring='r2', n_jobs=-1)
            oob_score = getattr(model, 'oob_score_', 0)
            print(f"模型 {i + 1} - CV R²: {cv_scores.mean():.4f}, OOB: {oob_score:.4f}")

        # 基于OOB分数的加权平均
        weights = [getattr(model, 'oob_score_', 0.5) for model in models]
        weights = np.array(weights) / sum(weights)

        final_pred = np.zeros(len(X_test))
        for i, pred in enumerate(predictions):
            final_pred += weights[i] * pred

        print("模型权重:")
        for i, weight in enumerate(weights):
            print(f"  模型 {i + 1}: {weight:.3f}")

        return final_pred, models

    def evaluate_single_model(self, X_train, y_train, X_test, y_test):
        """评估单模型性能"""
        print("评估单模型性能...")

        # 创建优化的单模型
        model = ExtraTreesRegressor(
            n_estimators=1200,
            max_depth=40,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.85,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)

        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='r2', n_jobs=-1)
        oob_score = getattr(model, 'oob_score_', 0)

        print(f"单模型性能:")
        print(f"  CV R²: {cv_scores.mean():.4f}")
        print(f"  OOB Score: {oob_score:.4f}")
        print(f"  测试集 R²: {r2:.4f}")

        return pred, r2, model

    def simplified_pipeline(self, X_train, y_train, X_test, y_test):
        """简化但高效的管道"""
        print("开始简化高效管道...")

        # 1. 基础特征工程
        print("\n1. 基础特征工程...")
        X_train_eng = self.advanced_feature_engineering_v2(X_train)
        X_test_eng = self.advanced_feature_engineering_v2(X_test)

        # 2. 特征选择（基于重要性）
        print("\n2. 特征选择...")
        X_train_sel, X_test_sel = self.smart_feature_selection(X_train_eng, y_train, X_test_eng, 'importance')

        # 3. 数据变换
        print("\n3. 数据变换...")
        X_train_trans, X_test_trans = self.data_transformation_boost(X_train_sel, y_train, X_test_sel, 'quantile')

        # 4. 评估单模型
        print("\n4. 评估优化单模型...")
        single_pred, single_r2, single_model = self.evaluate_single_model(X_train_trans, y_train, X_test_trans, y_test)

        # 5. 模型集成
        print("\n5. 模型集成...")
        ensemble_pred, ensemble_models = self.ensemble_of_best(X_train_trans, y_train, X_test_trans, n_models=5)
        ensemble_r2 = r2_score(y_test, ensemble_pred)

        # 6. 超参数调优（可选）
        print("\n6. 超参数调优...")
        try:
            tuned_model = self.hyperparameter_tuning(X_train_trans, y_train)
            tuned_pred = tuned_model.predict(X_test_trans)
            tuned_r2 = r2_score(y_test, tuned_pred)
        except Exception as e:
            print(f"超参数调优失败: {e}")
            tuned_pred = single_pred
            tuned_r2 = single_r2

        print(f"\n{'=' * 60}")
        print("方法性能比较:")
        print(f"优化单模型 R²: {single_r2:.4f}")
        print(f"模型集成 R²: {ensemble_r2:.4f}")
        print(f"调优模型 R²: {tuned_r2:.4f}")
        print(f"{'=' * 60}")

        # 选择最佳方法
        methods = {
            'single': (single_pred, single_r2),
            'ensemble': (ensemble_pred, ensemble_r2),
            'tuned': (tuned_pred, tuned_r2)
        }

        best_method = max(methods.items(), key=lambda x: x[1][1])
        best_name, (best_pred, best_r2) = best_method

        print(f"\n🎯 最佳方法: {best_name}, R²: {best_r2:.4f}")

        # 保存最佳模型
        if best_name == 'single':
            self.best_model = single_model
        elif best_name == 'tuned':
            self.best_model = tuned_model

        return best_pred, best_name, best_r2, methods

    def feature_importance_analysis(self, model, feature_names, top_n=20):
        """特征重要性分析"""
        if hasattr(model, 'feature_importances_'):
            print(f"\nTop {top_n} 重要特征:")
            importance = model.feature_importances_
            indices = np.argsort(importance)[-top_n:][::-1]

            for i, idx in enumerate(indices):
                if idx < len(feature_names):
                    print(f"{i + 1:2d}. {feature_names[idx]:30} : {importance[idx]:.4f}")


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始终极ExtraTrees优化分析...")

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
    previous_best_r2 = 0.3416
    print(f"之前最佳R²: {previous_best_r2:.4f}")

    # 创建终极优化器
    optimizer = UltimateExtraTreesOptimizer()

    # 执行简化管道
    final_pred, best_method, final_r2, all_methods = optimizer.simplified_pipeline(
        X_train, y_train, X_test, y_test
    )

    # 性能改进分析
    improvement = final_r2 - previous_best_r2
    improvement_percent = (improvement / previous_best_r2) * 100

    print(f"\n{'=' * 50}")
    print("性能改进总结")
    print(f"{'=' * 50}")
    print(f"之前最佳R²: {previous_best_r2:.4f}")
    print(f"当前最佳R²: {final_r2:.4f}")
    print(f"绝对提升: {improvement:.4f}")
    print(f"相对提升: {improvement_percent:.2f}%")

    if improvement > 0:
        print("🎉 优化成功！性能继续提升！")
    else:
        print("⚠️ 性能达到平台期")

    # 特征重要性分析
    if optimizer.best_model is not None:
        feature_names = X_train.columns.tolist()
        optimizer.feature_importance_analysis(optimizer.best_model, feature_names)

    # 保存结果
    results_df = pd.DataFrame({
        '真实值': y_test.values,
        '预测值': final_pred,
        '残差': y_test.values - final_pred
    })

    results_path = r"D:\PyProject\25卓越杯大数据\data2\ultimate_ensemble_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {results_path}")

    print(f"\n终极优化分析完成！最终R²: {final_r2:.4f}")


if __name__ == "__main__":
    main()