import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------------
# 1. 读取训练集 Parquet 数据
# -------------------------------
print("Loading training dataset from Parquet...")
df_train = pd.read_parquet("./train_em_cl.parquet")
print(f"Training dataset shape: {df_train.shape}")

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
# 3. 提取特征
# -------------------------------
feature_columns = [col for col in df_train.columns if col.startswith("embedding_dim_")]
if len(feature_columns) == 0:
    raise ValueError("❌ No embedding feature columns found!")

X_train = df_train[feature_columns].values
y_train = df_train[label_column].values

print(f"Training Features shape: {X_train.shape}")
print(f"Labels shape: {y_train.shape}")
print(f"Unique classes: {np.unique(y_train)}")

# -------------------------------
# 4. 【加速关键1】PCA 降维（例如降到 128 或 256 维）
# -------------------------------
n_components = 256  # 可选：64, 128, 256；推荐 128~256
print(f"\nApplying PCA to reduce dimension to {n_components}...")
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)

print(f"Reduced features shape: {X_train_pca.shape}")
# 保留 95%+ 信息？
print(f"Explained variance ratio (cumulative): {pca.explained_variance_ratio_.sum():.4f}")

# -------------------------------
# 5. 【加速关键2】标准化（SVM 必需）
# -------------------------------
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pca)
print("✅ Preprocessing completed.")

# -------------------------------
# 6. 【加速关键3】使用 SGDClassifier（线性 SVM，支持大规模数据）
# -------------------------------
print("\n✅ Training Linear SVM using SGD (fast, scalable)...")

svm_model = SGDClassifier(
    loss='hinge',           # 相当于线性 SVM
    alpha=0.0001,           # 正则化强度 (1/C)
    max_iter=1000,          # 可根据数据量调整
    tol=1e-4,
    random_state=42,
    n_jobs=-1,              # 并行
    learning_rate='optimal',
    verbose=0               # 可设为1查看训练进度
)

# 训练（速度极快）
svm_model.fit(X_train_scaled, y_train)
print("✅ Training completed.")

# -------------------------------
# 7. 加载测试集并评估
# -------------------------------
test_data_path = "./test_em_cl.parquet"
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"❌ Test file not found: {test_data_path}")

print(f"\nLoading test dataset: {test_data_path}")
df_test = pd.read_parquet(test_data_path)

if label_column not in df_test.columns:
    raise ValueError(f"❌ Label column '{label_column}' not found in test data.")

X_test = df_test[feature_columns].values
y_test = df_test[label_column].values

# 同样进行 PCA + 标准化
X_test_pca = pca.transform(X_test)          # ⚠️ 只 transform
X_test_scaled = scaler.transform(X_test_pca) # ⚠️ 使用训练集 scaler

print(f"Test set shape after PCA: {X_test_scaled.shape}")

# -------------------------------
# 8. 预测与评估
# -------------------------------
print("\nEvaluating on test set...")
y_pred = svm_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 9. 混淆矩阵可视化
# -------------------------------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix - Test Set (Fast SVM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# -------------------------------
# 10. 保存所有组件（模型 + scaler + PCA）
# -------------------------------
model_save_dir = "../models"
os.makedirs(model_save_dir, exist_ok=True)

joblib.dump(svm_model, os.path.join(model_save_dir, "svm_fast_protein_classifier.pkl"))
joblib.dump(scaler, os.path.join(model_save_dir, "feature_scaler.pkl"))
joblib.dump(pca, os.path.join(model_save_dir, "pca_transformer.pkl"))

print(f"\n💾 Model saved to '{model_save_dir}/svm_fast_protein_classifier.pkl'")
print(f"💾 Scaler saved.")
print(f"💾 PCA transformer saved.")