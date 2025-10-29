import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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


def build_svr_model(train_df, test_df, target_column='value'):
    """使用SVR建立模型"""

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
        return None

    # 3. 数据标准化（SVR对特征尺度敏感）
    print("\n正在进行数据标准化...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_original = y_test.copy()  # 保存原始y_test用于后续反标准化

    # 4. 使用常用参数创建SVR模型
    # 常用参数设置：
    # C: 1.0 (正则化参数)
    # epsilon: 0.1 (容忍度)
    # kernel: 'rbf' (径向基函数核，适合非线性问题)
    optimal_params = {'C': 1.0, 'epsilon': 0.01, 'kernel': 'rbf'}
    print(f"使用常用参数: {optimal_params}")

    # 5. 创建并训练SVR模型
    svr_model = SVR(
        C=optimal_params['C'],
        epsilon=optimal_params['epsilon'],
        kernel=optimal_params['kernel'],
        cache_size=1000  # 增大缓存以提高速度
    )

    print("\n开始训练SVR模型...")
    svr_model.fit(X_train_scaled, y_train_scaled)
    print("SVR模型训练完成！")

    # 6. 模型预测
    print("进行模型预测...")
    y_pred_scaled = svr_model.predict(X_test_scaled)

    # 将预测结果反标准化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 7. 模型评估
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    print("\n" + "=" * 50)
    print("SVR模型评估结果:")
    print(f"使用参数: {optimal_params}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print("=" * 50)

    return svr_model, X_train, X_test, y_train, y_test_original, y_pred, optimal_params, scaler_X, scaler_y


def plot_results(y_test, y_pred, model_name="SVR"):
    """绘制预测结果可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 真实值 vs 预测值散点图
    axes[0].scatter(y_test, y_pred, alpha=0.7, s=50)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title(f'{model_name} - 真实值 vs 预测值')
    axes[0].grid(True, alpha=0.3)

    # 添加R²到图中
    r2 = r2_score(y_test, y_pred)
    axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. 残差图
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7, s=50)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].set_title(f'{model_name} - 残差图')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return residuals


def feature_importance_analysis_svr(model, feature_names, scaler):
    """SVR特征重要性分析（仅适用于线性核）"""
    if model.kernel != 'linear':
        print(f"\n当前使用 {model.kernel} 核，无法直接获取特征重要性")
        print("特征重要性分析仅适用于线性核SVR")
        return None

    # 获取线性SVR的系数
    coefficients = model.coef_[0]

    # 由于数据被标准化，我们需要调整系数以反映原始特征的重要性
    # 系数需要除以特征的标准差
    feature_std = scaler.scale_
    adjusted_coefficients = coefficients / feature_std

    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'adjusted_coefficient': adjusted_coefficients,
        'abs_coefficient': np.abs(adjusted_coefficients)
    }).sort_values('abs_coefficient', ascending=False)

    print(f"\nSVR线性模型特征重要性分析:")
    print(f"特征数量: {len(feature_names)}")

    # 显示最重要的特征
    if len(importance) > 0:
        top_20_features = importance.head(20)
        print("\n前20个最重要的特征:")
        print(top_20_features)

        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_20_features)), top_20_features['abs_coefficient'])
        plt.yticks(range(len(top_20_features)), top_20_features['feature'])
        plt.xlabel('特征重要性（调整后系数绝对值）')
        plt.title('SVR特征重要性 - 前20个特征')
        plt.gca().invert_yaxis()

        # 在条形上添加数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                     ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()
    else:
        print("警告：无法计算特征重要性！")

    return importance


def analyze_data_distribution(train_df, test_df, target_column='value'):
    """分析训练集和测试集的分布"""
    print("\n" + "=" * 50)
    print("训练集和测试集分布分析")
    print("=" * 50)

    train_stats = train_df[target_column].describe()
    test_stats = test_df[target_column].describe()

    print(f"\n训练集 {target_column} 统计:")
    print(f"  均值: {train_stats['mean']:.4f}")
    print(f"  标准差: {train_stats['std']:.4f}")
    print(f"  最小值: {train_stats['min']:.4f}")
    print(f"  最大值: {train_stats['max']:.4f}")

    print(f"\n测试集 {target_column} 统计:")
    print(f"  均值: {test_stats['mean']:.4f}")
    print(f"  标准差: {test_stats['std']:.4f}")
    print(f"  最小值: {test_stats['min']:.4f}")
    print(f"  最大值: {test_stats['max']:.4f}")

    # 绘制分布对比图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(train_df[target_column], alpha=0.7, label='训练集', bins=15, density=True)
    plt.hist(test_df[target_column], alpha=0.7, label='测试集', bins=15, density=True)
    plt.xlabel(target_column)
    plt.ylabel('密度')
    plt.title('训练集 vs 测试集分布')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([train_df[target_column], test_df[target_column]],
                labels=['训练集', '测试集'])
    plt.title('训练集 vs 测试集箱线图')

    plt.tight_layout()
    plt.show()


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

        # 3. 建立SVR模型
        model_result = build_svr_model(train_df, test_df)

        if model_result is not None:
            svr_model, X_train, X_test, y_train, y_test, y_pred, optimal_params, scaler_X, scaler_y = model_result

            # 4. 可视化结果
            residuals = plot_results(y_test, y_pred, "SVR回归")

            # 5. 特征重要性分析（仅适用于线性核）
            feature_columns = [col for col in train_df.columns if col != 'value']
            importance_df = feature_importance_analysis_svr(svr_model, feature_columns, scaler_X)

            # 6. 保存模型和标准化器
            import joblib

            # 保存SVR模型
            model_path = r"D:\PyProject\25卓越杯大数据\data2\svr_model.pkl"
            joblib.dump(svr_model, model_path)
            print(f"\nSVR模型已保存到: {model_path}")

            # 保存标准化器
            scaler_X_path = r"D:\PyProject\25卓越杯大数据\data2\scaler_X.pkl"
            scaler_y_path = r"D:\PyProject\25卓越杯大数据\data2\scaler_y.pkl"
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)
            print(f"特征标准化器已保存到: {scaler_X_path}")
            print(f"目标变量标准化器已保存到: {scaler_y_path}")

            # 7. 显示预测结果对比
            results_df = pd.DataFrame({
                '真实值': y_test,
                '预测值': y_pred,
                '残差': residuals
            })
            print("\n测试集预测结果 (前10行):")
            print(results_df.head(10))

            # 8. 保存预测结果到CSV
            results_path = r"/data2/svr_result.csv"
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"SVR预测结果已保存到: {results_path}")

            print("\nSVR模型训练和评估完成！")