import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------------
# 1. 读取训练集 Parquet 数据（全量用于训练）
# -------------------------------
print("Loading training dataset from Parquet...")
df_train = pd.read_parquet("./train_em_cl.parquet")

print(f"Training dataset shape: {df_train.shape}")
print(f"Columns: {list(df_train.columns)}")

# -------------------------------
# 2. 确定标签列
# -------------------------------
label_column = "label"  # 请根据实际列名修改！
if label_column not in df_train.columns:
    possible_labels = ['Label', 'class', 'Class', 'y']
    for col in possible_labels:
        if col in df_train.columns:
            label_column = col
            break
    else:
        raise ValueError(f"❌ No label column found in training data. Available columns: {df_train.columns}")

# -------------------------------
# 3. 分离训练集特征和标签
# -------------------------------
feature_columns = [col for col in df_train.columns if col.startswith("embedding_dim_")]
if len(feature_columns) == 0:
    raise ValueError("❌ No embedding feature columns found! Check column name prefix.")

X_train = df_train[feature_columns].values
y_train = df_train[label_column].values

print(f"Training Features shape: {X_train.shape}")
print(f"Training Labels shape: {y_train.shape}")
print(f"Unique classes in training set: {np.unique(y_train)}")

# -------------------------------
# 4. 特征标准化（SVM 必需！）
# -------------------------------
print("\nStandardizing features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 拟合并转换训练数据

print("✅ Feature scaling completed.")

# -------------------------------
# 5. 训练 SVM 模型（使用全量训练数据）
# -------------------------------
print("\n✅ Training SVM Classifier (RBF kernel) on FULL training set...")

# 推荐参数（适用于蛋白质嵌入这类高维数据）
svm_model = SVC(
    kernel='rbf',           # RBF 核通常是最佳起点
    C=1.0,                  # 正则化强度，可后续调优
    gamma='scale',          # 自动适配特征尺度
    random_state=42,
    verbose=False,
    probability=False       # 若不需要预测概率，设为 False 提升速度
)

# 训练模型
svm_model.fit(X_train_scaled, y_train)
print("✅ SVM Training completed.")

# -------------------------------
# 6. 加载独立测试集并评估
# -------------------------------
test_data_path = "test_em_cl.parquet"
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"❌ Test file not found: {test_data_path}")

print(f"\nLoading test dataset from '{test_data_path}'...")
df_test = pd.read_parquet(test_data_path)
print(f"Test dataset shape: {df_test.shape}")

# 检查标签列
if label_column not in df_test.columns:
    raise ValueError(f"❌ Label column '{label_column}' not found in test data. Available: {df_test.columns}")

# 提取测试特征并标准化（使用训练集的 scaler）
X_test = df_test[feature_columns].values
y_test = df_test[label_column].values

X_test_scaled = scaler.transform(X_test)  # ⚠️ 只 transform，不 fit！

print(f"Test Features shape: {X_test.shape}")
print(f"Test Labels shape: {y_test.shape}")
print(f"Unique classes in test set: {np.unique(y_test)}")

# -------------------------------
# 7. 在测试集上预测和评估
# -------------------------------
print("\nEvaluating on test set...")
y_pred = svm_model.predict(X_test_scaled)

# 准确率
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# 分类报告
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 8. 混淆矩阵可视化
# -------------------------------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix - Test Set (SVM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# -------------------------------
# 9. 保存模型和 scaler（重要！推理时需要 scaler）
# -------------------------------
model_save_dir = "../models"
os.makedirs(model_save_dir, exist_ok=True)

# 保存 SVM 模型
model_save_path = os.path.join(model_save_dir, "svm_protein_classifier.pkl")
joblib.dump(svm_model, model_save_path)

# 保存 scaler（必须！否则未来无法预测）
scaler_save_path = os.path.join(model_save_dir, "feature_scaler.pkl")
joblib.dump(scaler, scaler_save_path)

print(f"\n💾 Model saved to '{model_save_path}'")
print(f"💾 Scaler saved to '{scaler_save_path}'")