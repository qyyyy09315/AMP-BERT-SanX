import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_separate_datasets(train_path, test_path):
    """分别加载训练集和测试集"""
    try:
        print("正在加载数据集...")
        train_df = pd.read_csv(train_path)
        print("训练集加载成功！")
        print(f"训练集形状: {train_df.shape}")

        test_df = pd.read_csv(test_path)
        print("测试集加载成功！")
        print(f"测试集形状: {test_df.shape}")

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


def optimize_individual_models(X_train, y_train):
    """优化Lasso和SVR模型"""
    print("开始优化单个模型...")

    # Lasso参数优化
    print("1. 优化Lasso模型...")
    lasso_param_grid = {
        'alpha': [0.001, 0.01, 0.1],
        'max_iter': [5000]
    }

    lasso_grid = GridSearchCV(
        Lasso(random_state=42),
        lasso_param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    lasso_grid.fit(X_train, y_train)
    best_lasso = lasso_grid.best_estimator_
    print(f"Lasso最优参数: {lasso_grid.best_params_}")
    print(f"Lasso交叉验证MSE: {-lasso_grid.best_score_:.4f}")

    # SVR参数优化
    print("2. 优化SVR模型...")
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.1],
        'kernel': ['rbf']
    }

    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    svr_grid = GridSearchCV(
        svr_pipeline,
        {'svr__' + key: value for key, value in svr_param_grid.items()},
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    svr_grid.fit(X_train, y_train)
    best_svr = svr_grid.best_estimator_
    print(f"SVR最优参数: {svr_grid.best_params_}")
    print(f"SVR交叉验证MSE: {-svr_grid.best_score_:.4f}")

    return best_lasso, best_svr


def build_weighted_ensemble(X_train, y_train, X_test, best_lasso, best_svr):
    """构建加权集成模型"""
    print("构建加权集成模型...")

    # 基于交叉验证性能计算权重
    model_weights = {}
    model_performances = {}

    print("计算模型权重...")

    # Lasso性能评估
    lasso_scores = cross_val_score(best_lasso, X_train, y_train,
                                   cv=3, scoring='neg_mean_squared_error')
    lasso_mse = -lasso_scores.mean()
    model_performances['lasso'] = lasso_mse
    model_weights['lasso'] = 1.0 / lasso_mse

    # SVR性能评估
    svr_scores = cross_val_score(best_svr, X_train, y_train,
                                 cv=3, scoring='neg_mean_squared_error')
    svr_mse = -svr_scores.mean()
    model_performances['svr'] = svr_mse
    model_weights['svr'] = 1.0 / svr_mse

    # 归一化权重
    total_weight = sum(model_weights.values())
    for name in model_weights:
        model_weights[name] /= total_weight

    print("模型权重分配:")
    for name, weight in model_weights.items():
        print(f"  {name}: {weight:.3f} (MSE: {model_performances[name]:.4f})")

    # 加权集成预测
    lasso_pred = best_lasso.predict(X_test)
    svr_pred = best_svr.predict(X_test)
    y_pred = (model_weights['lasso'] * lasso_pred + model_weights['svr'] * svr_pred)

    return y_pred, model_weights, lasso_pred, svr_pred


def evaluate_model(y_test, y_pred, model_name="加权集成模型"):
    """评估模型性能"""
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"{model_name}评估结果:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print("=" * 60)

    return mse, rmse, mae, r2


def plot_weighted_ensemble_results(y_test, lasso_pred, svr_pred, weighted_pred, weights):
    """绘制加权集成模型结果"""
    print("绘制结果图表...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 三个模型的预测对比
    models_preds = [lasso_pred, svr_pred, weighted_pred]
    model_names = ['Lasso', 'SVR', '加权集成']
    colors = ['blue', 'green', 'red']

    for i, (pred, name, color) in enumerate(zip(models_preds, model_names, colors)):
        axes[0, 0].scatter(y_test, pred, alpha=0.6, label=name, color=color, s=30)
        r2 = r2_score(y_test, pred)
        axes[0, 0].text(0.05, 0.95 - i * 0.1, f'{name} R² = {r2:.3f}',
                        transform=axes[0, 0].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.8)
    axes[0, 0].set_xlabel('真实值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title('模型预测对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 残差对比
    residuals = [y_test - lasso_pred, y_test - svr_pred, y_test - weighted_pred]
    for i, (residual, name, color) in enumerate(zip(residuals, model_names, colors)):
        axes[0, 1].scatter(models_preds[i], residual, alpha=0.6, label=name, color=color, s=30)

    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('预测值')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 权重可视化
    names = list(weights.keys())
    values = list(weights.values())
    bars = axes[1, 0].bar(names, values, color=['blue', 'green'], alpha=0.7)
    axes[1, 0].set_ylabel('权重')
    axes[1, 0].set_title('模型权重分配')
    axes[1, 0].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{value:.3f}', ha='center', va='bottom')

    # 4. 性能对比
    metrics = ['MSE', 'MAE', 'R²']
    lasso_metrics = [
        mean_squared_error(y_test, lasso_pred),
        mean_absolute_error(y_test, lasso_pred),
        r2_score(y_test, lasso_pred)
    ]
    svr_metrics = [
        mean_squared_error(y_test, svr_pred),
        mean_absolute_error(y_test, svr_pred),
        r2_score(y_test, svr_pred)
    ]
    weighted_metrics = [
        mean_squared_error(y_test, weighted_pred),
        mean_absolute_error(y_test, weighted_pred),
        r2_score(y_test, weighted_pred)
    ]

    x = np.arange(len(metrics))
    width = 0.25

    axes[1, 1].bar(x - width, lasso_metrics, width, label='Lasso', alpha=0.8)
    axes[1, 1].bar(x, svr_metrics, width, label='SVR', alpha=0.8)
    axes[1, 1].bar(x + width, weighted_metrics, width, label='加权集成', alpha=0.8)

    axes[1, 1].set_xlabel('评估指标')
    axes[1, 1].set_ylabel('数值')
    axes[1, 1].set_title('模型性能对比')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    # 文件路径
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    print("开始加权集成(Lasso+SVR)模型分析...")

    # 1. 加载数据
    train_df, test_df = load_separate_datasets(train_path, test_path)
    if train_df is None or test_df is None:
        return

    # 2. 统一特征名
    train_df, test_df, success = unify_feature_names(train_df, test_df, 'value')
    if not success:
        return

    # 3. 准备数据
    feature_columns = [col for col in train_df.columns if col != 'value']
    X_train = train_df[feature_columns]
    y_train = train_df['value']
    X_test = test_df[feature_columns]
    y_test = test_df['value']

    print(f"特征数量: {len(feature_columns)}")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 4. 优化单个模型
    best_lasso, best_svr = optimize_individual_models(X_train, y_train)

    # 5. 构建加权集成模型
    weighted_pred, weights, lasso_pred, svr_pred = build_weighted_ensemble(
        X_train, y_train, X_test, best_lasso, best_svr
    )

    # 6. 评估加权集成模型
    mse, rmse, mae, r2 = evaluate_model(y_test, weighted_pred, "加权集成(Lasso+SVR)")

    # 7. 与单个模型对比
    lasso_r2 = r2_score(y_test, lasso_pred)
    svr_r2 = r2_score(y_test, svr_pred)

    print(f"模型对比分析:")
    print(f"Lasso单独 R²: {lasso_r2:.4f}")
    print(f"SVR单独 R²: {svr_r2:.4f}")
    print(f"加权集成 R²: {r2:.4f}")

    improvement = r2 - max(lasso_r2, svr_r2)
    print(f"集成提升: {improvement:.4f}")

    if improvement > 0:
        print("加权集成有效提升了模型性能！")
    else:
        print("加权集成未能提升性能")

    # 8. 绘制结果
    plot_weighted_ensemble_results(y_test, lasso_pred, svr_pred, weighted_pred, weights)

    # 9. 保存结果
    import joblib

    # 保存模型权重信息
    model_info = {
        'lasso_model': best_lasso,
        'svr_model': best_svr,
        'weights': weights,
        'feature_columns': feature_columns
    }

    model_path = r"D:\PyProject\25卓越杯大数据\data2\weighted_ensemble_model.pkl"
    joblib.dump(model_info, model_path)
    print(f"模型已保存到: {model_path}")

    # 保存预测结果
    results_df = pd.DataFrame({
        '真实值': y_test,
        'Lasso预测': lasso_pred,
        'SVR预测': svr_pred,
        '加权集成预测': weighted_pred,
        '集成残差': y_test - weighted_pred
    })

    results_path = r"D:\PyProject\25卓越杯大数据\data2\weighted_ensemble_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存到: {results_path}")

    # 保存评估结果
    evaluation_df = pd.DataFrame({
        '模型': ['Lasso', 'SVR', '加权集成'],
        'MSE': [
            mean_squared_error(y_test, lasso_pred),
            mean_squared_error(y_test, svr_pred),
            mse
        ],
        'R²': [lasso_r2, svr_r2, r2],
        '权重': [weights['lasso'], weights['svr'], 1.0]
    })

    eval_path = r"D:\PyProject\25卓越杯大数据\data2\model_evaluation.csv"
    evaluation_df.to_csv(eval_path, index=False, encoding='utf-8-sig')
    print(f"评估结果已保存到: {eval_path}")

    print(f"加权集成(Lasso+SVR)分析完成！")


if __name__ == "__main__":
    main()