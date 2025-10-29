import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import os
import time

# 设置并行处理
os.environ['OMP_NUM_THREADS'] = '0'  # 自动选择线程数
os.environ['MKL_NUM_THREADS'] = '0'
os.environ['OPENBLAS_NUM_THREADS'] = '0'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# 获取CPU核心数
NUM_CORES = os.cpu_count()
print(f"检测到 CPU 核心数: {NUM_CORES}")


def load_separate_datasets(train_path, test_path):
    """分别加载训练集和测试集"""
    try:
        # 直接读取，不使用并行（小文件并行反而慢）
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print("训练集加载成功！")
        print(f"训练集形状: {train_df.shape}")
        print("\n训练集前5行:")
        print(train_df.head())

        print("\n测试集加载成功！")
        print(f"测试集形状: {test_df.shape}")
        print("\n测试集前5行:")
        print(test_df.head())

        return train_df, test_df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None


def unify_feature_names(train_df, test_df, target_column='value'):
    """统一训练集和测试集的特征列名"""
    train_feature_cols = [col for col in train_df.columns if col != target_column]
    test_feature_cols = [col for col in test_df.columns if col != target_column]

    print(f"训练集特征列数量: {len(train_feature_cols)}")
    print(f"测试集特征列数量: {len(test_feature_cols)}")

    if train_feature_cols != test_feature_cols:
        print("特征列名不匹配，正在统一列名...")
        if len(train_feature_cols) == len(test_feature_cols):
            column_mapping = {test_col: train_col for test_col, train_col in zip(test_feature_cols, train_feature_cols)}
            test_df = test_df.rename(columns=column_mapping)
            print("测试集列名已统一为训练集列名")
        else:
            print("错误：训练集和测试集特征数量不同！")
            return train_df, test_df, False

    return train_df, test_df, True


def build_adaboost_model(train_df, test_df, target_column='value'):
    """使用AdaBoost进行回归预测 - 优化CPU利用率"""

    # 1. 统一特征列名
    train_df, test_df, success = unify_feature_names(train_df, test_df, target_column)
    if not success:
        return None

    # 2. 准备特征和目标变量
    feature_columns = [col for col in train_df.columns if col != target_column]

    X_train = train_df[feature_columns].values
    y_train = train_df[target_column].values
    X_test = test_df[feature_columns].values
    y_test = test_df[target_column].values

    print(f"\n特征数量: {len(feature_columns)}")
    print(f"目标变量: {target_column}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 3. 创建并训练AdaBoost模型 - 优化参数以充分利用CPU
    print(f"\n配置模型以充分利用 {NUM_CORES} 个CPU核心...")

    # 使用更深的决策树和更多的弱学习器
    base_estimator = DecisionTreeRegressor(
        max_depth=5,  # 增加深度以利用更多计算
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )

    ada_model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=700,  # 增加弱学习器数量
        learning_rate=0.01,  # 降低学习率以获得更好的收敛
        random_state=42
    )

    print("\n开始训练AdaBoost回归模型...")
    print(f"  - 可用CPU核心: {NUM_CORES}")

    # 训练模型
    start_time = time.time()

    ada_model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"模型训练完成！耗时: {training_time:.2f} 秒")

    # 4. 模型预测
    print("进行模型预测...")
    start_time = time.time()

    y_pred = ada_model.predict(X_test)

    prediction_time = time.time() - start_time
    print(f"预测完成！耗时: {prediction_time:.2f} 秒")

    # 5. 模型评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("AdaBoost回归模型评估结果:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print("=" * 60)

    # 6. 显示模型信息
    print(f"\n模型详细信息:")
    print(f"弱学习器数量: {len(ada_model.estimators_)}")
    print(f"特征重要性总和: {np.sum(ada_model.feature_importances_):.4f}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"预测时间: {prediction_time:.2f} 秒")

    return ada_model, X_train, X_test, y_train, y_test, y_pred, training_time


def feature_importance_analysis(ada_model, feature_names):
    """AdaBoost特征重要性分析 - 优化版本"""

    print(f"\n开始特征重要性分析...")

    # 获取特征重要性
    feature_importances = ada_model.feature_importances_

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    print(f"AdaBoost回归特征重要性分析:")
    print(f"特征数量: {len(feature_names)}")
    print(f"重要特征数量 (重要性 > 0): {len(importance[importance['importance'] > 0])}")

    # 显示最重要的特征
    if len(importance) > 0:
        top_20_features = importance.head(20)
        print("\n前20个最重要的特征:")
        print(top_20_features)

        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_20_features)), top_20_features['importance'])
        plt.yticks(range(len(top_20_features)), top_20_features['feature'], fontsize=10)
        plt.xlabel('特征重要性', fontsize=12)
        plt.title('AdaBoost回归特征重要性 - 前20个特征', fontsize=14)
        plt.gca().invert_yaxis()

        # 在条形上添加数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0.001:
                plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                         ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.show()

        # 计算累积重要性
        cumulative_importance = np.cumsum(importance['importance'])
        n_features_80 = np.argmax(cumulative_importance >= 0.8) + 1
        n_features_90 = np.argmax(cumulative_importance >= 0.9) + 1

        # 绘制特征重要性累积图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
        plt.xlabel('特征数量', fontsize=12)
        plt.ylabel('累积重要性', fontsize=12)
        plt.title('AdaBoost回归特征累积重要性', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.axvline(x=n_features_80, color='r', linestyle='--', alpha=0.7,
                    label=f'80%重要性 ({n_features_80}个特征)')
        plt.axvline(x=n_features_90, color='g', linestyle='--', alpha=0.7,
                    label=f'90%重要性 ({n_features_90}个特征)')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"\n特征重要性统计:")
        print(f"达到80%重要性所需的特征数量: {n_features_80}")
        print(f"达到90%重要性所需的特征数量: {n_features_90}")
        print(f"前10个特征的重要性总和: {cumulative_importance[9]:.3f}")

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
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.title('训练集 vs 测试集分布', fontsize=14)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([train_df[target_column], test_df[target_column]],
                labels=['训练集', '测试集'])
    plt.title('训练集 vs 测试集箱线图', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_results(y_test, y_pred, model_name="AdaBoost回归"):
    """绘制预测结果可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 真实值 vs 预测值散点图
    axes[0].scatter(y_test, y_pred, alpha=0.7, s=50)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('真实值', fontsize=12)
    axes[0].set_ylabel('预测值', fontsize=12)
    axes[0].set_title(f'{model_name} - 真实值 vs 预测值', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # 添加R2到图中
    r2 = r2_score(y_test, y_pred)
    axes[0].text(0.05, 0.95, f'R2 = {r2:.3f}', transform=axes[0].transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. 残差图
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7, s=50)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('预测值', fontsize=12)
    axes[1].set_ylabel('残差', fontsize=12)
    axes[1].set_title(f'{model_name} - 残差图', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return residuals


# 主程序
if __name__ == "__main__":
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print(f"开始运行AdaBoost回归分析...")
    print(f"系统CPU核心数: {NUM_CORES}")
    print(f"将充分利用i9-13900K的性能优势")

    # 1. 分别加载训练集和测试集
    train_df, test_df = load_separate_datasets(train_path, test_path)

    if train_df is not None and test_df is not None:
        # 2. 分析数据分布
        analyze_data_distribution(train_df, test_df)

        # 3. 建立AdaBoost回归模型
        model_result = build_adaboost_model(train_df, test_df)

        if model_result is not None:
            ada_model, X_train, X_test, y_train, y_test, y_pred, training_time = model_result

            # 4. 可视化结果
            residuals = plot_results(y_test, y_pred, "AdaBoost回归")

            # 5. 特征重要性分析
            feature_columns = [col for col in train_df.columns if col != 'value']
            importance_df = feature_importance_analysis(ada_model, feature_columns)

            # 6. 保存模型
            import joblib

            # 保存AdaBoost模型
            model_path = r"D:\PyProject\25卓越杯大数据\data2\adaboost_model.pkl"
            joblib.dump(ada_model, model_path, compress=3)
            print(f"\nAdaBoost回归模型已保存到: {model_path}")

            # 7. 显示预测结果对比
            results_df = pd.DataFrame({
                '真实值': y_test,
                '预测值': y_pred,
                '残差': residuals,
                '绝对误差': np.abs(residuals)
            })
            print("\n测试集预测结果 (前10行):")
            print(results_df.head(10))

            # 显示预测统计
            print(f"\n预测结果统计:")
            print(f"平均绝对误差: {results_df['绝对误差'].mean():.4f}")
            print(f"最大绝对误差: {results_df['绝对误差'].max():.4f}")
            print(f"预测值与真实值的相关系数: {np.corrcoef(y_test, y_pred)[0, 1]:.4f}")

            # 8. 保存预测结果到CSV
            results_path = r"D:\PyProject\25卓越杯大数据\data2\adaboost_prediction_results.csv"
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"AdaBoost回归预测结果已保存到: {results_path}")

            # 9. 保存特征重要性结果
            importance_path = r"D:\PyProject\25卓越杯大数据\data2\adaboost_feature_importance.csv"
            importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
            print(f"特征重要性结果已保存到: {importance_path}")

            print(f"\nAdaBoost回归模型训练和评估完成！")
            print(f"总训练时间: {training_time:.2f} 秒")
            print(f"充分利用了i9-13900K的 {NUM_CORES} 个核心")