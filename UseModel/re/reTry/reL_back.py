import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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


def build_linear_regression_model_separate(train_df, test_df, target_column='value'):
    """使用分开的数据集建立线性回归模型"""

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

    # 3. 创建并训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. 模型预测
    y_pred = model.predict(X_test)

    # 5. 模型评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("模型评估结果:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print("=" * 50)

    # 6. 显示模型系数
    print(f"\n截距 (Intercept): {model.intercept_:.4f}")

    # 只显示前20个最重要的特征系数
    coefficients = pd.DataFrame({
        '特征': feature_columns,
        '系数': model.coef_
    })
    coefficients['系数绝对值'] = np.abs(coefficients['系数'])
    top_20_features = coefficients.nlargest(20, '系数绝对值')
    print("\n前20个最重要的特征系数:")
    print(top_20_features)

    return model, X_train, X_test, y_train, y_test, y_pred


def plot_results(y_test, y_pred):
    """绘制预测结果可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 真实值 vs 预测值散点图
    axes[0].scatter(y_test, y_pred, alpha=0.7)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('真实值')
    axes[0].set_ylabel('预测值')
    axes[0].set_title('真实值 vs 预测值')
    axes[0].grid(True, alpha=0.3)

    # 2. 残差图
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('预测值')
    axes[1].set_ylabel('残差')
    axes[1].set_title('残差图')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return residuals


def feature_importance_analysis(model, feature_names):
    """特征重要性分析"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    # 只显示前30个最重要的特征
    top_30 = importance.head(30)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_30, x='abs_coefficient', y='feature')
    plt.title('特征重要性（基于系数绝对值）- 前30个特征')
    plt.xlabel('系数绝对值')
    plt.tight_layout()
    plt.show()

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
                tick_labels=['训练集', '测试集'])  # 修复这里
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

        # 3. 建立线性回归模型
        model_result = build_linear_regression_model_separate(train_df, test_df)

        if model_result is not None:
            model, X_train, X_test, y_train, y_test, y_pred = model_result

            # 4. 可视化结果
            residuals = plot_results(y_test, y_pred)

            # 5. 特征重要性分析
            feature_columns = [col for col in train_df.columns if col != 'value']
            importance_df = feature_importance_analysis(model, feature_columns)
            print("\n特征重要性排序 (前20个):")
            print(importance_df.head(20))

            # 6. 保存模型
            import joblib

            model_path = r"D:\PyProject\25卓越杯大数据\data2\linear_regression_model_separate.pkl"
            joblib.dump(model, model_path)
            print(f"\n模型已保存到: {model_path}")

            # 7. 显示预测结果对比
            results_df = pd.DataFrame({
                '真实值': y_test,
                '预测值': y_pred,
                '残差': residuals
            })
            print("\n测试集预测结果 (前10行):")
            print(results_df.head(10))

            # 8. 保存预测结果到CSV
            results_path = r"D:\PyProject\25卓越杯大数据\data2\back_result.csv"
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"预测结果已保存到: {results_path}")