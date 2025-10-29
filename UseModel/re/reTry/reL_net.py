import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_separate_datasets(train_path, test_path):
    """分别加载训练集和测试集"""
    try:
        # 读取训练集
        train_df = pd.read_csv(train_path)
        print("训练集加载成功！")
        print(f"训练集形状: {train_df.shape}")
        print("\n训练集前5行:")
        print(train_df.head())

        # 读取测试集
        test_df = pd.read_csv(test_path)
        print("\n测试集加载成功！")
        print(f"测试集形状: {test_df.shape}")
        print("\n测试集前5行:")
        print(test_df.head())

        # 检查数据基本信息
        print("\n训练集基本信息:")
        print(train_df.info())
        print("\n测试集基本信息:")
        print(test_df.info())

        print("\n训练集描述性统计:")
        print(train_df.describe())
        print("\n测试集描述性统计:")
        print(test_df.describe())

        return train_df, test_df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None


def unify_feature_names(train_df, test_df, target_column='value'):
    """统一训练集和测试集的特征列名"""

    # 获取训练集的特征列（排除目标列）
    train_feature_cols = [col for col in train_df.columns if col != target_column]

    # 获取测试集的特征列（排除目标列）
    test_feature_cols = [col for col in test_df.columns if col != target_column]

    print(f"训练集特征列数量: {len(train_feature_cols)}")
    print(f"测试集特征列数量: {len(test_feature_cols)}")

    # 检查列名是否匹配
    if train_feature_cols != test_feature_cols:
        print("特征列名不匹配，正在统一列名...")

        # 如果特征数量相同，只是列名不同，则重命名测试集列名
        if len(train_feature_cols) == len(test_feature_cols):
            # 创建列名映射
            column_mapping = {test_col: train_col for test_col, train_col in zip(test_feature_cols, train_feature_cols)}
            print(f"列名映射: {column_mapping}")

            # 重命名测试集列名
            test_df = test_df.rename(columns=column_mapping)
            print("测试集列名已统一为训练集列名")
        else:
            print("错误：训练集和测试集特征数量不同！")
            return train_df, test_df, False

    return train_df, test_df, True


def find_optimal_parameters(X_train, y_train):
    """使用交叉验证寻找最优的alpha和l1_ratio参数"""
    print("\n正在寻找最优的弹性网络参数...")

    # 定义参数范围
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # l1_ratio=1就是Lasso，=0就是Ridge

    # 使用网格搜索寻找最优参数
    elastic_net = ElasticNet(max_iter=10000, random_state=42)
    param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}

    grid_search = GridSearchCV(
        elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    best_l1_ratio = grid_search.best_params_['l1_ratio']
    best_score = -grid_search.best_score_

    print(f"最优alpha: {best_alpha}")
    print(f"最优l1_ratio: {best_l1_ratio}")
    print(f"最优MSE: {best_score:.4f}")

    # 显示参数搜索热力图
    results = pd.DataFrame(grid_search.cv_results_)
    scores = results.pivot_table(index='param_alpha', columns='param_l1_ratio',
                                 values='mean_test_score', aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(-scores, annot=True, fmt='.3f', cmap='viridis')
    plt.title('弹性网络参数调优热力图 (MSE)')
    plt.xlabel('L1 Ratio')
    plt.ylabel('Alpha')
    plt.show()

    return best_alpha, best_l1_ratio


def build_elastic_net_model(train_df, test_df, target_column='value', use_grid_search=True):
    """使用弹性网络回归建立模型"""

    # 1. 统一特征列名
    train_df, test_df, success = unify_feature_names(train_df, test_df, target_column)
    if not success:
        return None

    # 2. 准备特征和目标变量
    feature_columns = [col for col in train_df.columns if col != target_column]

    # 训练集
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    # 测试集
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    print(f"\n特征数量: {len(feature_columns)}")
    print(f"目标变量: {target_column}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 检查特征是否一致
    if list(X_train.columns) != list(X_test.columns):
        print("错误：训练集和测试集特征列仍然不一致！")
        print(f"训练集列: {list(X_train.columns)[:10]}...")
        print(f"测试集列: {list(X_test.columns)[:10]}...")
        return None

    # 3. 寻找最优参数或使用默认值
    if use_grid_search and len(feature_columns) > 1:
        optimal_alpha, optimal_l1_ratio = find_optimal_parameters(X_train, y_train)
    else:
        optimal_alpha = 1.0  # 默认值
        optimal_l1_ratio = 0.5  # 默认值，平衡L1和L2正则化
        print(f"使用默认参数: alpha={optimal_alpha}, l1_ratio={optimal_l1_ratio}")

    # 4. 创建并训练弹性网络模型
    elastic_net_model = ElasticNet(
        alpha=optimal_alpha,
        l1_ratio=optimal_l1_ratio,
        max_iter=10000,
        random_state=42
    )
    elastic_net_model.fit(X_train, y_train)

    # 5. 模型预测
    y_pred = elastic_net_model.predict(X_test)

    # 6. 模型评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("弹性网络回归模型评估结果:")
    print(f"最优Alpha参数: {optimal_alpha}")
    print(f"最优L1 Ratio: {optimal_l1_ratio}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print("=" * 50)

    # 7. 显示模型系数和稀疏性
    print(f"\n截距 (Intercept): {elastic_net_model.intercept_:.4f}")

    # 统计非零系数的特征数量
    non_zero_coef = np.sum(elastic_net_model.coef_ != 0)
    zero_coef = np.sum(elastic_net_model.coef_ == 0)
    total_coef = len(elastic_net_model.coef_)

    print(f"\n系数稀疏性分析:")
    print(f"非零系数特征数量: {non_zero_coef}/{total_coef} ({non_zero_coef / total_coef * 100:.1f}%)")
    print(f"零系数特征数量: {zero_coef}/{total_coef} ({zero_coef / total_coef * 100:.1f}%)")

    # 显示非零系数的特征
    coefficients = pd.DataFrame({
        '特征': feature_columns,
        '系数': elastic_net_model.coef_
    })
    non_zero_coefficients = coefficients[coefficients['系数'] != 0].copy()
    non_zero_coefficients['系数绝对值'] = np.abs(non_zero_coefficients['系数'])
    non_zero_coefficients = non_zero_coefficients.sort_values('系数绝对值', ascending=False)

    print(f"\n非零系数特征 ({len(non_zero_coefficients)} 个):")
    print(non_zero_coefficients)

    return elastic_net_model, X_train, X_test, y_train, y_test, y_pred, optimal_alpha, optimal_l1_ratio


def plot_results(y_test, y_pred, model_name="弹性网络回归"):
    """绘制预测结果可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 真实值 vs 预测值散点图
    axes[0].scatter(y_test, y_pred, alpha=0.7)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title(f'{model_name} - 真实值 vs 预测值')
    axes[0].grid(True, alpha=0.3)

    # 2. 残差图
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].set_title(f'{model_name} - 残差图')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return residuals


def feature_importance_analysis(model, feature_names, alpha_value, l1_ratio):
    """弹性网络特征重要性分析"""
    # 只考虑非零系数
    non_zero_mask = model.coef_ != 0
    non_zero_features = np.array(feature_names)[non_zero_mask]
    non_zero_coefficients = model.coef_[non_zero_mask]

    importance = pd.DataFrame({
        'feature': non_zero_features,
        'coefficient': non_zero_coefficients,
        'abs_coefficient': np.abs(non_zero_coefficients)
    }).sort_values('abs_coefficient', ascending=False)

    print(f"\n弹性网络模型 (alpha={alpha_value}, l1_ratio={l1_ratio}) 特征选择结果:")
    print(f"原始特征数量: {len(feature_names)}")
    print(f"选择后的特征数量: {len(importance)}")

    # 显示所有非零特征
    if len(importance) > 0:
        print("\n所有非零系数特征:")
        print(importance)

        # 绘制特征重要性图（最多显示30个）
        plot_features = importance.head(min(30, len(importance)))

        plt.figure(figsize=(12, 8))
        sns.barplot(data=plot_features, x='abs_coefficient', y='feature')
        plt.title(f'弹性网络特征重要性 (Alpha={alpha_value}, L1 Ratio={l1_ratio}) - 前{len(plot_features)}个特征')
        plt.xlabel('系数绝对值')
        plt.tight_layout()
        plt.show()
    else:
        print("警告：所有特征系数都为零！")

    return importance


def analyze_data_distribution(train_df, test_df, target_column='value'):
    """分析训练集和测试集的分布"""
    print("\n" + "=" * 50)
    print("训练集和测试集分布分析")
    print("=" * 50)

    print(f"\n训练集 {target_column} 统计:")
    print(f"  均值: {train_df[target_column].mean():.4f}")
    print(f"  标准差: {train_df[target_column].std():.4f}")
    print(f"  最小值: {train_df[target_column].min():.4f}")
    print(f"  最大值: {train_df[target_column].max():.4f}")

    print(f"\n测试集 {target_column} 统计:")
    print(f"  均值: {test_df[target_column].mean():.4f}")
    print(f"  标准差: {test_df[target_column].std():.4f}")
    print(f"  最小值: {test_df[target_column].min():.4f}")
    print(f"  最大值: {test_df[target_column].max():.4f}")

    # 绘制分布对比图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(train_df[target_column], alpha=0.7, label='训练集', bins=10)
    plt.hist(test_df[target_column], alpha=0.7, label='测试集', bins=10)
    plt.xlabel(target_column)
    plt.ylabel('频数')
    plt.title('训练集 vs 测试集分布')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([train_df[target_column], test_df[target_column]],
                labels=['训练集', '测试集'])
    plt.title('训练集 vs 测试集箱线图')

    plt.tight_layout()
    plt.show()


def compare_with_other_models(train_df, test_df, target_column='value'):
    """与Lasso和Ridge回归对比"""
    from sklearn.linear_model import Lasso, Ridge

    feature_columns = [col for col in train_df.columns if col != target_column]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    # Lasso回归 (l1_ratio=1.0)
    lasso_model = Lasso(alpha=1.0, max_iter=10000, random_state=42)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    lasso_mse = mean_squared_error(y_test, y_pred_lasso)
    lasso_non_zero = np.sum(lasso_model.coef_ != 0)

    # Ridge回归 (l1_ratio=0.0)
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, y_pred_ridge)
    ridge_non_zero = np.sum(ridge_model.coef_ != 0)

    print("\n" + "=" * 50)
    print("模型对比: 弹性网络 vs Lasso vs Ridge")
    print("=" * 50)
    print(f"Lasso回归 (L1正则化):")
    print(f"  MSE: {lasso_mse:.4f}")
    print(f"  非零系数特征: {lasso_non_zero}/{len(feature_columns)}")
    print(f"Ridge回归 (L2正则化):")
    print(f"  MSE: {ridge_mse:.4f}")
    print(f"  非零系数特征: {ridge_non_zero}/{len(feature_columns)}")


# 主程序
if __name__ == "__main__":
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    # 1. 分别加载训练集和测试集
    train_df, test_df = load_separate_datasets(train_path, test_path)

    if train_df is not None and test_df is not None:
        # 2. 分析数据分布
        analyze_data_distribution(train_df, test_df)

        # 3. 与其他模型对比
        compare_with_other_models(train_df, test_df)

        # 4. 建立弹性网络回归模型
        model_result = build_elastic_net_model(train_df, test_df, use_grid_search=True)

        if model_result is not None:
            elastic_net_model, X_train, X_test, y_train, y_test, y_pred, optimal_alpha, optimal_l1_ratio = model_result

            # 5. 可视化结果
            residuals = plot_results(y_test, y_pred, "弹性网络回归")

            # 6. 特征重要性分析
            feature_columns = [col for col in train_df.columns if col != 'value']
            importance_df = feature_importance_analysis(elastic_net_model, feature_columns, optimal_alpha,
                                                        optimal_l1_ratio)

            # 7. 保存模型
            import joblib

            model_path = r"D:\PyProject\25卓越杯大数据\data2\elastic_net_model.pkl"
            joblib.dump(elastic_net_model, model_path)
            print(f"\n弹性网络模型已保存到: {model_path}")

            # 8. 显示预测结果对比
            results_df = pd.DataFrame({
                '真实值': y_test,
                '预测值': y_pred,
                '残差': residuals
            })
            print("\n测试集预测结果 (前10行):")
            print(results_df.head(10))

            # 9. 保存预测结果到CSV
            results_path = r"D:\PyProject\25卓越杯大数据\data2\elastic_net_result.csv"
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"弹性网络预测结果已保存到: {results_path}")

