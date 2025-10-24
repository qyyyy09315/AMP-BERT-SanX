import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
# 2. 确定标签列（请根据你的数据调整）
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
# 4. 训练随机森林模型（使用全量训练数据）
# -------------------------------
print("\n✅ Training Random Forest Classifier on FULL training set...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

# 训练模型
rf_model.fit(X_train, y_train)
print("✅ Training completed.")

# -------------------------------
# 5. 加载独立测试集并评估
# -------------------------------
test_data_path = "test_em_cl.parquet"
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"❌ Test file not found: {test_data_path}")

print(f"\nLoading test dataset from '{test_data_path}'...")
df_test = pd.read_parquet(test_data_path)
print(f"Test dataset shape: {df_test.shape}")

# 检查测试集中是否有标签列
if label_column not in df_test.columns:
    raise ValueError(f"❌ Label column '{label_column}' not found in test data. Available: {df_test.columns}")

# 提取测试集特征和标签
X_test = df_test[feature_columns].values
y_test = df_test[label_column].values

print(f"Test Features shape: {X_test.shape}")
print(f"Test Labels shape: {y_test.shape}")
print(f"Unique classes in test set: {np.unique(y_test)}")

# -------------------------------
# 6. 在测试集上预测和评估
# -------------------------------
print("\nEvaluating on test set...")
y_pred = rf_model.predict(X_test)

# 准确率
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# 分类报告
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. 混淆矩阵可视化
# -------------------------------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# -------------------------------
# 8. 特征重要性可视化（前20个）
# -------------------------------
top_k = 20
importance = rf_model.feature_importances_
top_indices = np.argsort(importance)[::-1][:top_k]
top_importance = importance[top_indices]
top_names = [f"dim_{i}" for i in top_indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importance, y=top_names, palette="viridis")
plt.title(f"Top {top_k} Important Embedding Dimensions")
plt.xlabel("Feature Importance")
plt.ylabel("Dimension")
plt.tight_layout()
plt.show()

# -------------------------------
# 9. 保存模型
# -------------------------------
model_save_dir = "../models"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "rf_protein_classifier.pkl")

joblib.dump(rf_model, model_save_path)
print(f"\n💾 Model saved to '{model_save_path}'")