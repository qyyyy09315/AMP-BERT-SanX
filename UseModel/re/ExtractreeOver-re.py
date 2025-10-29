import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FinalPerformanceAnalyzer:
    def __init__(self):
        self.optimization_history = {
            '起始模型': 0.3279,
            '基础优化': 0.3416,
            '中级优化': 0.3458,
            '高级优化': 0.3539,
            '终极优化': 0.3553
        }

    def create_comprehensive_summary(self):
        """创建全面的优化总结"""
        print("=" * 80)
        print("ExtraTrees模型优化历程完整总结")
        print("=" * 80)

        stages = list(self.optimization_history.keys())
        scores = list(self.optimization_history.values())

        # 计算各阶段提升
        improvements = [0]
        improvement_percents = [0]

        for i in range(1, len(scores)):
            abs_improvement = scores[i] - scores[i - 1]
            rel_improvement = (abs_improvement / scores[i - 1]) * 100
            improvements.append(abs_improvement)
            improvement_percents.append(rel_improvement)

        total_improvement = scores[-1] - scores[0]
        total_improvement_pct = (total_improvement / scores[0]) * 100

        print(f"{'优化阶段':<15} {'R²分数':<10} {'绝对提升':<12} {'相对提升':<12} {'累积提升':<12}")
        print("-" * 80)

        cumulative_improvement = 0
        for i, (stage, score, imp, imp_pct) in enumerate(zip(stages, scores, improvements, improvement_percents)):
            cumulative_improvement += imp
            if i == 0:
                print(f"{stage:<15} {score:<10.4f} {'-':<12} {'-':<12} {'-':<12}")
            else:
                print(f"{stage:<15} {score:<10.4f} {imp:<12.4f} {imp_pct:<11.2f}% {cumulative_improvement:<12.4f}")

        print("-" * 80)
        print(
            f"{'总提升':<15} {'':<10} {total_improvement:<12.4f} {total_improvement_pct:<11.2f}% {total_improvement:<12.4f}")
        print("=" * 80)

        return stages, scores, improvements, improvement_percents

    def plot_optimization_journey(self, stages, scores, improvements):
        """绘制优化历程图"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 1. R²进展图
        ax1.plot(stages, scores, 'o-', linewidth=3, markersize=8, color='steelblue', markerfacecolor='red')
        ax1.set_title('ExtraTrees模型R²优化历程', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_xlabel('优化阶段', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 在点上添加数值
        for i, (stage, score) in enumerate(zip(stages, scores)):
            ax1.annotate(f'{score:.4f}', (stage, score),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # 2. 各阶段提升幅度
        improvement_stages = stages[1:]
        colors = ['lightgreen', 'lightblue', 'lightcoral', 'gold']

        bars = ax2.bar(improvement_stages, improvements[1:], color=colors, alpha=0.8)
        ax2.set_title('各阶段性能提升幅度', fontsize=14, fontweight='bold')
        ax2.set_ylabel('R²提升', fontsize=12)
        ax2.set_xlabel('优化阶段', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值
        for bar, imp in zip(bars, improvements[1:]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.0001,
                     f'+{imp:.4f}', ha='center', va='bottom', fontweight='bold')

        # 3. 累积提升图
        cumulative_improvements = np.cumsum(improvements)
        ax3.plot(stages, cumulative_improvements, 's-', linewidth=2, markersize=8, color='purple')
        ax3.fill_between(stages, cumulative_improvements, alpha=0.3, color='purple')
        ax3.set_title('累积性能提升', fontsize=14, fontweight='bold')
        ax3.set_ylabel('累积R²提升', fontsize=12)
        ax3.set_xlabel('优化阶段', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 在点上添加数值
        for i, (stage, cum_imp) in enumerate(zip(stages, cumulative_improvements)):
            ax3.annotate(f'{cum_imp:.4f}', (stage, cum_imp),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def analyze_final_performance(self, y_true, y_pred):
        """分析最终模型性能"""
        print("\n" + "=" * 50)
        print("最终模型性能详细分析")
        print("=" * 50)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")

        # 残差分析
        residuals = y_true - y_pred
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75)
        }

        print(f"\n残差统计分析:")
        print(f"  均值: {residual_stats['mean']:.4f}")
        print(f"  标准差: {residual_stats['std']:.4f}")
        print(f"  范围: [{residual_stats['min']:.4f}, {residual_stats['max']:.4f}]")
        print(f"  25%分位数: {residual_stats['q25']:.4f}")
        print(f"  75%分位数: {residual_stats['q75']:.4f}")

        # 预测准确度分析
        absolute_errors = np.abs(residuals)
        error_percentage = np.abs(residuals / y_true) * 100
        error_percentage = error_percentage[np.isfinite(error_percentage)]

        print(f"\n预测准确度:")
        print(f"  平均绝对误差: {np.mean(absolute_errors):.4f}")
        print(f"  平均相对误差: {np.mean(error_percentage):.2f}%")

        # 误差分布统计
        error_thresholds = [0.1, 0.2, 0.3, 0.5]
        print(f"\n绝对误差分布:")
        for threshold in error_thresholds:
            within_threshold = np.sum(absolute_errors <= threshold) / len(absolute_errors) * 100
            print(f"  误差 ≤ {threshold}: {within_threshold:.1f}%")

        return {
            'metrics': {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2},
            'residual_stats': residual_stats,
            'accuracy': {
                'mean_absolute_error': np.mean(absolute_errors),
                'mean_relative_error': np.mean(error_percentage)
            }
        }

    def suggest_next_optimization_directions(self, current_r2=0.3553):
        """建议下一步优化方向"""
        print("\n" + "=" * 60)
        print("下一步优化方向建议")
        print("=" * 60)

        print("基于当前R² = 0.3553，建议尝试以下方向:")

        directions = [
            {
                '方向': '深度特征工程',
                '具体方法': ['自动特征生成', '领域知识特征', '时间序列特征', '图特征'],
                '预期提升': '0.002-0.005',
                '难度': '中等'
            },
            {
                '方向': '高级集成方法',
                '具体方法': ['深度学习集成', '自动机器学习', '元学习', '多任务学习'],
                '预期提升': '0.003-0.006',
                '难度': '高'
            },
            {
                '方向': '数据质量优化',
                '具体方法': ['异常值检测', '数据清洗', '样本权重', '数据增强'],
                '预期提升': '0.001-0.003',
                '难度': '低'
            },
            {
                '方向': '模型架构创新',
                '具体方法': ['注意力机制', 'Transformer', '图神经网络', '强化学习'],
                '预期提升': '0.004-0.008',
                '难度': '高'
            }
        ]

        for i, direction in enumerate(directions, 1):
            print(f"\n{i}. {direction['方向']}:")
            print(f"   方法: {', '.join(direction['具体方法'])}")
            print(f"   预期提升: R² +{direction['预期提升']}")
            print(f"   难度: {direction['难度']}")

        print(f"\n总体预期: 通过系统优化，R²有望达到 0.36-0.37")
        print("=" * 60)

    def save_final_report(self, analysis_results, file_path):
        """保存最终报告"""
        report = {
            'optimization_summary': {
                'initial_r2': 0.3279,
                'final_r2': 0.3553,
                'total_improvement': 0.0274,
                'relative_improvement': 8.36
            },
            'performance_metrics': analysis_results['metrics'],
            'residual_analysis': analysis_results['residual_stats'],
            'accuracy_analysis': analysis_results['accuracy'],
            'key_techniques_used': [
                "智能特征工程和选择",
                "超参数优化",
                "多样性模型集成",
                "残差提升技术",
                "数据变换和预处理"
            ],
            'optimization_timeline': self.optimization_history
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n完整分析报告已保存到: {file_path}")


def main():
    """主分析函数"""
    # 加载最终结果
    results_path = r"D:\PyProject\25卓越杯大数据\data2\ultimate_optimization_results.csv"
    results_df = pd.read_csv(results_path)

    y_true = results_df['真实值'].values
    y_pred = results_df['预测值'].values

    print("开始最终性能综合分析...")

    # 创建分析器
    analyzer = FinalPerformanceAnalyzer()

    # 1. 优化历程总结
    stages, scores, improvements, improvement_percents = analyzer.create_comprehensive_summary()

    # 2. 绘制优化历程图
    analyzer.plot_optimization_journey(stages, scores, improvements)

    # 3. 详细性能分析
    analysis_results = analyzer.analyze_final_performance(y_true, y_pred)

    # 4. 建议下一步方向
    analyzer.suggest_next_optimization_directions()

    # 5. 保存最终报告
    report_path = r"D:\PyProject\25卓越杯大数据\data2\final_optimization_report.json"
    analyzer.save_final_report(analysis_results, report_path)

    print("\n" + "=" * 80)
    print("🎉 ExtraTrees模型优化项目圆满完成！")
    print("=" * 80)
    print(f"📈 总性能提升: 0.0274 (相对提升: 8.36%)")
    print(f"🏆 最终R²分数: 0.3553")
    print(f"🚀 从起始0.3279到最终0.3553，实现了显著的性能提升！")
    print(f"💡 建议继续探索深度特征工程和高级集成方法")
    print("=" * 80)


if __name__ == "__main__":
    main()