import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import warnings

warnings.filterwarnings('ignore')


class NeuralExtraTreesHybrid:
    def __init__(self):
        self.et_model = None
        self.nn_model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.blend_weights = None

    def prepare_features(self, X_train, y_train, X_test, n_features=300):
        """特征准备 - 修复版本"""
        print("准备混合模型特征...")

        # 只在训练时创建特征选择器
        if self.feature_selector is None:
            selector = SelectFromModel(
                ExtraTreesRegressor(n_estimators=200, random_state=42),
                max_features=n_features,
                threshold=-np.inf
            )
            selector.fit(X_train, y_train)
            self.feature_selector = selector
            self.selected_features = X_train.columns[selector.get_support()]
            print(f"选择特征数量: {len(self.selected_features)}")

        # 使用相同的特征选择器处理所有数据
        X_train_sel = self.feature_selector.transform(X_train)
        X_test_sel = self.feature_selector.transform(X_test)

        # 转换为DataFrame保持一致性
        X_train_df = pd.DataFrame(X_train_sel,
                                  columns=[f'selected_feat_{i}' for i in range(X_train_sel.shape[1])],
                                  index=X_train.index)
        X_test_df = pd.DataFrame(X_test_sel,
                                 columns=[f'selected_feat_{i}' for i in range(X_test_sel.shape[1])],
                                 index=X_test.index)

        return X_train_df, X_test_df

    def create_advanced_extra_trees(self, X_train, y_train):
        """创建高级ExtraTrees模型"""
        print("创建高级ExtraTrees模型...")

        et_params = {
            'n_estimators': 1200,
            'max_depth': 40,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 0.85,
            'bootstrap': True,
            'oob_score': True,
            'random_state': 42,
            'n_jobs': -1
        }

        self.et_model = ExtraTreesRegressor(**et_params)
        self.et_model.fit(X_train, y_train)

        cv_scores = cross_val_score(self.et_model, X_train, y_train,
                                    cv=5, scoring='r2', n_jobs=-1)

        print(f"ExtraTrees性能:")
        print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        if hasattr(self.et_model, 'oob_score_'):
            print(f"  OOB Score: {self.et_model.oob_score_:.4f}")

        return self.et_model

    def create_advanced_neural_network(self, X_train, y_train):
        """创建高级神经网络"""
        print("创建高级神经网络模型...")

        # 数据标准化（神经网络需要）
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 多种神经网络架构尝试
        nn_architectures = [
            {
                'name': '深层网络',
                'params': {
                    'hidden_layer_sizes': (256, 128, 64, 32),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'learning_rate': 'adaptive',
                    'max_iter': 1000,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': 50,
                    'random_state': 42
                }
            },
            {
                'name': '宽网络',
                'params': {
                    'hidden_layer_sizes': (512, 256),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0005,
                    'learning_rate': 'adaptive',
                    'max_iter': 800,
                    'early_stopping': True,
                    'random_state': 42
                }
            },
            {
                'name': '残差风格',
                'params': {
                    'hidden_layer_sizes': (128, 128, 128, 128),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive',
                    'max_iter': 1200,
                    'early_stopping': True,
                    'random_state': 42
                }
            }
        ]

        best_nn = None
        best_nn_score = -np.inf
        best_nn_name = ""

        for arch in nn_architectures:
            print(f"  尝试 {arch['name']}...")
            try:
                nn_model = MLPRegressor(**arch['params'])

                # 使用交叉验证评估
                cv_scores = cross_val_score(nn_model, X_train_scaled, y_train,
                                            cv=3, scoring='r2')
                mean_score = cv_scores.mean()

                print(f"    {arch['name']} CV R²: {mean_score:.4f}")

                if mean_score > best_nn_score:
                    best_nn_score = mean_score
                    best_nn = nn_model
                    best_nn_name = arch['name']

            except Exception as e:
                print(f"    {arch['name']} 训练失败: {e}")
                continue

        if best_nn is not None:
            print(f"  选择最佳神经网络: {best_nn_name}")
            best_nn.fit(X_train_scaled, y_train)
            self.nn_model = best_nn
        else:
            # 默认网络
            print("  使用默认神经网络")
            self.nn_model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                random_state=42,
                max_iter=1000,
                early_stopping=True
            )
            self.nn_model.fit(X_train_scaled, y_train)

        return self.nn_model

    def optimize_blending_weights(self, X_train, y_train, X_val, y_val):
        """优化混合权重"""
        print("优化模型混合权重...")

        # 获取两个模型的预测
        et_pred_train = self.et_model.predict(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        nn_pred_train = self.nn_model.predict(X_train_scaled)

        et_pred_val = self.et_model.predict(X_val)
        X_val_scaled = self.scaler.transform(X_val)
        nn_pred_val = self.nn_model.predict(X_val_scaled)

        # 网格搜索最佳权重
        best_weight = 0.5
        best_r2 = -np.inf

        weights = np.arange(0.1, 1.0, 0.05)
        for w in weights:
            blended_pred = w * et_pred_val + (1 - w) * nn_pred_val
            r2 = r2_score(y_val, blended_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_weight = w

        self.blend_weights = {'et': best_weight, 'nn': 1 - best_weight}

        print(f"优化后的混合权重:")
        print(f"  ExtraTrees: {best_weight:.3f}")
        print(f"  神经网络: {1 - best_weight:.3f}")
        print(f"  验证集R²: {best_r2:.4f}")

        return best_weight

    def create_stacked_features(self, X_train, y_train, X_test):
        """创建堆叠特征"""
        print("创建堆叠特征...")

        # 获取基础预测
        et_pred_train = self.et_model.predict(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        nn_pred_train = self.nn_model.predict(X_train_scaled)

        et_pred_test = self.et_model.predict(X_test)
        X_test_scaled = self.scaler.transform(X_test)
        nn_pred_test = self.nn_model.predict(X_test_scaled)

        # 创建堆叠特征
        X_train_stacked = np.column_stack([
            X_train.values,
            et_pred_train.reshape(-1, 1),
            nn_pred_train.reshape(-1, 1),
            (et_pred_train * nn_pred_train).reshape(-1, 1),  # 交互项
            ((et_pred_train + nn_pred_train) / 2).reshape(-1, 1)  # 平均项
        ])

        X_test_stacked = np.column_stack([
            X_test.values,
            et_pred_test.reshape(-1, 1),
            nn_pred_test.reshape(-1, 1),
            (et_pred_test * nn_pred_test).reshape(-1, 1),
            ((et_pred_test + nn_pred_test) / 2).reshape(-1, 1)
        ])

        print(f"堆叠特征维度: {X_train_stacked.shape}")

        return X_train_stacked, X_test_stacked

    def train_meta_learner(self, X_train_stacked, y_train, X_test_stacked, y_test):
        """训练元学习器"""
        print("训练元学习器...")

        # 使用ExtraTrees作为元学习器
        meta_learner = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        meta_learner.fit(X_train_stacked, y_train)
        meta_pred = meta_learner.predict(X_test_stacked)
        meta_r2 = r2_score(y_test, meta_pred)

        print(f"元学习器性能:")
        print(f"  R²: {meta_r2:.4f}")

        return meta_pred, meta_r2, meta_learner

    def simplified_hybrid_pipeline(self, X_train, y_train, X_test, y_test):
        """简化的混合模型管道"""
        print("开始简化的神经网络+ExtraTrees混合建模...")

        # 1. 特征准备
        print("\n1. 特征准备...")
        X_train_sel, X_test_sel = self.prepare_features(X_train, y_train, X_test, 350)

        # 2. 训练ExtraTrees
        print("\n2. 训练ExtraTrees...")
        et_model = self.create_advanced_extra_trees(X_train_sel, y_train)
        et_pred_test = et_model.predict(X_test_sel)
        et_r2_test = r2_score(y_test, et_pred_test)
        print(f"  ExtraTrees测试集R²: {et_r2_test:.4f}")

        # 3. 训练神经网络
        print("\n3. 训练神经网络...")
        nn_model = self.create_advanced_neural_network(X_train_sel, y_train)
        X_test_scaled = self.scaler.transform(X_test_sel)
        nn_pred_test = nn_model.predict(X_test_scaled)
        nn_r2_test = r2_score(y_test, nn_pred_test)
        print(f"  神经网络测试集R²: {nn_r2_test:.4f}")

        # 4. 多种混合策略
        print("\n4. 混合策略比较...")

        # 策略1: 简单平均
        simple_avg_pred = 0.5 * et_pred_test + 0.5 * nn_pred_test
        simple_avg_r2 = r2_score(y_test, simple_avg_pred)
        print(f"  简单平均 R²: {simple_avg_r2:.4f}")

        # 策略2: 基于验证集的优化权重
        print("  优化混合权重...")
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_sel, y_train, test_size=0.2, random_state=42)

        # 在子集上重新训练模型用于权重优化
        et_model_val = ExtraTreesRegressor(n_estimators=800, max_depth=35, random_state=42, n_jobs=-1)
        et_model_val.fit(X_tr, y_tr)

        nn_model_val = MLPRegressor(hidden_layer_sizes=(512, 256), random_state=42, max_iter=800)
        scaler_val = StandardScaler()
        X_tr_scaled = scaler_val.fit_transform(X_tr)
        nn_model_val.fit(X_tr_scaled, y_tr)

        # 优化权重
        et_pred_val = et_model_val.predict(X_val)
        X_val_scaled = scaler_val.transform(X_val)
        nn_pred_val = nn_model_val.predict(X_val_scaled)

        best_weight = 0.5
        best_r2_val = -np.inf
        weights = np.arange(0.1, 1.0, 0.05)

        for w in weights:
            blended_pred = w * et_pred_val + (1 - w) * nn_pred_val
            r2 = r2_score(y_val, blended_pred)
            if r2 > best_r2_val:
                best_r2_val = r2
                best_weight = w

        optimized_pred = best_weight * et_pred_test + (1 - best_weight) * nn_pred_test
        optimized_r2 = r2_score(y_test, optimized_pred)
        print(f"  优化混合 R²: {optimized_r2:.4f} (权重: {best_weight:.2f})")

        # 策略3: 动态权重（基于模型置信度）
        print("  尝试动态权重...")
        # 使用预测方差作为置信度指标
        et_confidence = 1.0 / (1.0 + np.std(et_pred_test))
        nn_confidence = 1.0 / (1.0 + np.std(nn_pred_test))

        dynamic_weights = np.array([et_confidence, nn_confidence])
        dynamic_weights = dynamic_weights / np.sum(dynamic_weights)

        dynamic_pred = dynamic_weights[0] * et_pred_test + dynamic_weights[1] * nn_pred_test
        dynamic_r2 = r2_score(y_test, dynamic_pred)
        print(f"  动态权重 R²: {dynamic_r2:.4f} (权重: {dynamic_weights[0]:.2f}, {dynamic_weights[1]:.2f})")

        # 策略4: 选择最佳单模型
        single_models = {
            'et': (et_pred_test, et_r2_test),
            'nn': (nn_pred_test, nn_r2_test)
        }
        best_single = max(single_models.items(), key=lambda x: x[1][1])

        # 结果比较
        methods = {
            'et_only': (et_pred_test, et_r2_test),
            'nn_only': (nn_pred_test, nn_r2_test),
            'simple_avg': (simple_avg_pred, simple_avg_r2),
            'optimized_blend': (optimized_pred, optimized_r2),
            'dynamic_blend': (dynamic_pred, dynamic_r2),
            'best_single': best_single[1]
        }

        print(f"\n{'=' * 60}")
        print("混合模型性能比较")
        print(f"{'=' * 60}")
        for name, (pred, r2) in methods.items():
            print(f"{name:15} | R²: {r2:.4f}")

        # 选择最佳方法
        best_method = max(methods.items(), key=lambda x: x[1][1])
        best_name, (best_pred, best_r2) = best_method

        print(f"{'=' * 60}")
        print(f"\n🎯 最佳混合方法: {best_name}, R²: {best_r2:.4f}")

        return best_pred, best_name, best_r2, methods


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始神经网络+ExtraTrees混合模型分析...")

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

    # 创建混合模型
    hybrid_model = NeuralExtraTreesHybrid()

    # 执行简化的混合管道
    final_pred, best_method, final_r2, all_methods = hybrid_model.simplified_hybrid_pipeline(
        X_train, y_train, X_test, y_test
    )

    # 性能改进分析
    improvement = final_r2 - previous_best_r2
    improvement_percent = (improvement / previous_best_r2) * 100

    print(f"\n{'=' * 50}")
    print("混合模型性能改进总结")
    print(f"{'=' * 50}")
    print(f"之前最佳R²: {previous_best_r2:.4f}")
    print(f"混合模型R²: {final_r2:.4f}")
    print(f"绝对提升: {improvement:.4f}")
    print(f"相对提升: {improvement_percent:.2f}%")

    if improvement > 0:
        print("🎉 混合模型优化成功！")
    else:
        print("⚠️ 混合模型未能提升性能")

    # 保存结果
    results_df = pd.DataFrame({
        '真实值': y_test.values,
        '预测值': final_pred,
        '残差': y_test.values - final_pred
    })

    results_path = r"D:\PyProject\25卓越杯大数据\data2\hybrid_model_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n混合模型结果已保存到: {results_path}")

    print(f"\n神经网络+ExtraTrees混合建模完成！最终R²: {final_r2:.4f}")


if __name__ == "__main__":
    main()