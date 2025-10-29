import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

torch.manual_seed(42)
np.random.seed(42)


# 定义残差块（适配更大的 hidden_size）
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_size, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, in_features)  # 保持输入输出维度一致
        self.bn2 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(dropout_rate)

        # 如果输入输出维度不一致（比如特征变换），可用 shortcut 调整
        # 但此处 in_features == hidden_size 或 in_features == out_features，无需调整
        self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out


# 深度残差网络模型（加深加宽）
class DeepResNet(nn.Module):
    def __init__(self, input_size, block=ResidualBlock, num_blocks=8, hidden_size=256, dropout_rate=0.5):
        super(DeepResNet, self).__init__()

        # 初始层：将输入映射到高维空间
        self.initial = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # 堆叠多个残差块
        layers = []
        for _ in range(num_blocks):
            layers.append(block(hidden_size, hidden_size, dropout_rate))
        self.res_blocks = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Linear(hidden_size, 1)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x.squeeze()


def load_separate_datasets(train_path, test_path):
    """分别加载训练集和测试集"""
    try:
        train_df = pd.read_csv(train_path)
        print("训练集加载成功！")
        print(f"训练集形状: {train_df.shape}")
        print("\n训练集前5行:")
        print(train_df.head())

        test_df = pd.read_csv(test_path)
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


def build_residual_deep_learning_model(train_df, test_df, target_column='value'):
    """使用更深更宽的深度残差网络建立模型"""

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

    # 3. 数据标准化
    print("\n正在进行数据标准化...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # 4. 创建训练集和验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )

    print(f"训练分割集大小: {X_train_split.shape}")
    print(f"验证分割集大小: {X_val_split.shape}")

    # 5. 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_split)
    y_train_tensor = torch.FloatTensor(y_train_split)
    X_val_tensor = torch.FloatTensor(X_val_split)
    y_val_tensor = torch.FloatTensor(y_val_split)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 增大 batch_size

    # 6. 构建更深更宽的深度残差网络
    input_size = X_train_scaled.shape[1]
    model = DeepResNet(
        input_size=input_size,
        num_blocks=8,           # 加深：从6 → 8个残差块
        hidden_size=256,        # 加宽：从128 → 256神经元
        dropout_rate=0.5
    )

    print(f"\n深度残差网络结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,} (可训练: {trainable_params:,})")

    # 7. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)  # 更小学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    # 8. 训练模型
    print("\n开始训练深度残差网络...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    num_epochs = 1500  # 增加训练轮数
    patience = 100     # 更大耐心
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_loss = criterion(val_outputs, y_val_tensor.to(device)).item()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停判断
        if val_loss < best_loss * 0.998:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
        else:
            patience_counter += 1

        if epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}, Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'\n早停触发！在 epoch {epoch} 停止训练')
            print(f'最佳验证损失: {best_loss:.6f} (epoch {best_epoch})')
            break
    else:
        print(f'\n训练完成！达到最大轮次 {num_epochs}')
        if best_model_state is None:
            best_model_state = model.state_dict().copy()
            best_epoch = num_epochs - 1

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 9. 在测试集上评估
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor.to(device)).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 10. 模型评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("深度残差网络模型评估结果:")
    print(f"训练轮次: {best_epoch + 1}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print("=" * 60)

    # 绘制训练曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='验证损失', alpha=0.8, linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'最佳模型 (epoch {best_epoch})')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.title('训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='训练损失', alpha=0.8, linewidth=2)
    plt.plot(val_losses, label='验证损失', alpha=0.8, linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    plt.xlabel('训练轮次')
    plt.ylabel('损失 (对数尺度)')
    plt.title('训练过程 (对数尺度)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    last_n = 100
    start_idx = max(0, len(train_losses) - last_n)
    epochs_range = range(start_idx, len(train_losses))
    plt.plot(epochs_range, train_losses[start_idx:], label='训练损失', alpha=0.8, linewidth=2)
    plt.plot(epochs_range, val_losses[start_idx:], label='验证损失', alpha=0.8, linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'最佳模型')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.title(f'训练过程 (最后 {last_n} 轮)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return model, X_train, X_test, y_train, y_test, y_pred, scaler_X, scaler_y, train_losses, val_losses, best_epoch


def plot_results(y_test, y_pred, model_name="深度残差网络"):
    """绘制预测结果可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 真实 vs 预测
    axes[0].scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('真实值', fontsize=12)
    axes[0].set_ylabel('预测值', fontsize=12)
    axes[0].set_title(f'{model_name} - 真实值 vs 预测值', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    r2 = r2_score(y_test, y_pred)
    axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 残差图
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=50, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('预测值', fontsize=12)
    axes[1].set_ylabel('残差', fontsize=12)
    axes[1].set_title(f'{model_name} - 残差图', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return residuals


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

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(train_df[target_column], alpha=0.7, label='训练集', bins=20, density=True)
    plt.hist(test_df[target_column], alpha=0.7, label='测试集', bins=20, density=True)
    plt.xlabel(target_column)
    plt.ylabel('密度')
    plt.title('目标变量分布对比')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([train_df[target_column], test_df[target_column]],
                labels=['训练集', '测试集'])
    plt.title('箱线图对比')
    plt.ylabel(target_column)

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 文件路径（请根据实际路径修改）
    train_path = r"D:\PyProject\25卓越杯大数据\data2\train_re_with_delete.csv"
    test_path = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"

    # 1. 加载数据
    train_df, test_df = load_separate_datasets(train_path, test_path)

    if train_df is not None and test_df is not None:
        # 2. 分布分析
        analyze_data_distribution(train_df, test_df)

        # 3. 训练深度残差网络（更深更宽）
        model_result = build_residual_deep_learning_model(train_df, test_df)

        if model_result is not None:
            model, X_train, X_test, y_train, y_test, y_pred, scaler_X, scaler_y, train_losses, val_losses, best_epoch = model_result

            # 4. 可视化结果
            residuals = plot_results(y_test, y_pred, "深度残差网络")

            # 5. 保存模型和标准化器
            model_path = r"D:\PyProject\25卓越杯大数据\data2\deep_residual_nn_model_enhanced.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': X_train.shape[1],
                'num_blocks': 8,
                'hidden_size': 256,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, model_path)
            print(f"\n深度残差网络模型已保存到: {model_path}")

            scaler_X_path = r"D:\PyProject\25卓越杯大数据\data2\scaler_X_res_enhanced.pkl"
            scaler_y_path = r"D:\PyProject\25卓越杯大数据\data2\scaler_y_res_enhanced.pkl"
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)
            print(f"特征标准化器已保存到: {scaler_X_path}")
            print(f"目标变量标准化器已保存到: {scaler_y_path}")

            # 6. 显示预测结果
            results_df = pd.DataFrame({
                '真实值': y_test,
                '预测值': y_pred,
                '残差': residuals,
                '绝对误差': np.abs(residuals)
            })
            print("\n测试集预测结果 (前10行):")
            print(results_df.head(10))

            print(f"\n预测结果统计:")
            print(f"平均绝对误差: {results_df['绝对误差'].mean():.4f}")
            print(f"最大绝对误差: {results_df['绝对误差'].max():.4f}")
            print(f"相关系数: {np.corrcoef(y_test, y_pred)[0, 1]:.4f}")

            # 7. 保存预测结果
            results_path = r"D:\PyProject\25卓越杯大数据\data2\residual_nn_prediction_results_enhanced.csv"
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"预测结果已保存到: {results_path}")

            print("\n✅ 深度残差网络模型（加深加宽）训练和评估完成！")