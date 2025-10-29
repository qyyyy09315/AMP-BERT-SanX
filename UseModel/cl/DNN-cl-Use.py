import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from copy import deepcopy

# -------------------------------
# 1. 读取训练集 Parquet 数据
# -------------------------------
print("Loading training dataset from Parquet...")
df_train = pd.read_parquet("./train_em_cl.parquet")
print(f"Training dataset shape: {df_train.shape}")
print(f"Columns: {list(df_train.columns)}")

# -------------------------------
# 2. 确定标签列
# -------------------------------
label_column = "label"
if label_column not in df_train.columns:
    possible_labels = ['Label', 'class', 'Class', 'y']
    for col in possible_labels:
        if col in df_train.columns:
            label_column = col
            break
    else:
        raise ValueError(f"❌ No label column found. Available: {df_train.columns}")

# -------------------------------
# 3. 提取特征和标签
# -------------------------------
feature_columns = [col for col in df_train.columns if col.startswith("embedding_dim_")]
if len(feature_columns) == 0:
    raise ValueError("❌ No embedding feature columns found!")

X_train = df_train[feature_columns].values
y_train = df_train[label_column].values

print(f"Training Features shape: {X_train.shape}")
print(f"Training Labels shape: {y_train.shape}")
print(f"Unique classes in training set: {np.unique(y_train)}")

# -------------------------------
# 4. 标准化特征（推荐用于 DNN）
# -------------------------------
print("Standardizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------------
# 5. 加载测试集并标准化
# -------------------------------
test_data_path = "test_em_cl.parquet"
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"❌ Test file not found: {test_data_path}")

print(f"\nLoading test dataset: {test_data_path}")
df_test = pd.read_parquet(test_data_path)

if label_column not in df_test.columns:
    raise ValueError(f"❌ Label column '{label_column}' not found in test data.")

X_test = df_test[feature_columns].values
y_test = df_test[label_column].values

X_test_scaled = scaler.transform(X_test)  # 只 transform

print(f"Test set shape: {X_test.shape}")
print(f"Unique classes in test set: {np.unique(y_test)}")

# -------------------------------
# 6. 转换为 PyTorch Tensors
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)

X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# -------------------------------
# 7. 构建 DNN 模型
# -------------------------------
class ProteinDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout=0.3):
        super(ProteinDNN, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 加速训练
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 实例化模型
input_dim = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train))
hidden_dims = [1024, 512, 256, 128]  # 可调整

model = ProteinDNN(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, dropout=0.3)
model.to(device)

print(f"\nModel Architecture:\n{model}")

# -------------------------------
# 8. 训练配置
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Adam + L2 正则
batch_size = 64
num_epochs = 1000  # 足够大，靠早停终止
patience = 50     # 早停耐心值

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# 9. 早停机制 + 训练循环
# -------------------------------
print("\n✅ Starting DNN training with Early Stopping...")

best_loss = float('inf')
best_epoch = 0
best_model_wts = None
patience_counter = 0

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # --- 验证阶段（使用测试集作为验证集）
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        val_loss = criterion(outputs, y_test_tensor).item()
        _, preds = torch.max(outputs, 1)
        val_acc = accuracy_score(y_test, preds.cpu().numpy())

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # --- 早停判断 ---
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        best_model_wts = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    # 打印进度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Best at: {best_epoch+1}")

    # 判断是否提前停止
    if patience_counter >= patience:
        print(f"\n✅ Early stopping at epoch {epoch+1}")
        break

# 恢复最佳模型权重
model.load_state_dict(best_model_wts)
print(f"✅ Best model restored from epoch {best_epoch + 1}")

# -------------------------------
# 10. 在测试集上最终预测
# -------------------------------
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred = torch.max(outputs, 1)
    y_pred = y_pred.cpu().numpy()

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Test Accuracy: {acc:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 11. 混淆矩阵可视化
# -------------------------------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix - Test Set (DNN)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# -------------------------------
# 12. 训练过程可视化
# -------------------------------
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.axvline(best_epoch + 1, color='r', linestyle='--', label=f'Best Model (epoch {best_epoch+1})')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracies, label='Val Accuracy', color='g')
plt.axvline(best_epoch + 1, color='r', linestyle='--', label=f'Best Model')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# 13. 保存模型和 scaler
# -------------------------------
model_save_dir = "../../models"
os.makedirs(model_save_dir, exist_ok=True)

# 保存模型结构和权重
model_path = os.path.join(model_save_dir, "dnn_protein_classifier.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'num_classes': num_classes,
    'hidden_dims': hidden_dims,
    'feature_columns': feature_columns
}, model_path)

# 保存 scaler
scaler_path = os.path.join(model_save_dir, "feature_scaler.pkl")
joblib.dump(scaler, scaler_path)

print(f"\n💾 Model saved to '{model_path}'")
print(f"💾 Scaler saved to '{scaler_path}'")