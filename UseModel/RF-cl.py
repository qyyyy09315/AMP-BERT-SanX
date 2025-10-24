import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1. 读取 Parquet 数据
# -------------------------------
print("Loading dataset from Parquet...")
df = pd.read_parquet("DataAfterBert/train_em_cl.parquet")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# -------------------------------
# 2. 确定标签列（请根据你的数据调整）
# -------------------------------
# 👉 常见的标签列名：'label', 'Label', 'class', 'Class', 'y' 等
# 如果不确定，可以打印 df.columns 查看
label_column = "label"  # 请根据实际列名修改！

if label_column not in df.columns:
    # 尝试常见名称
    possible_labels = ['Label', 'class', 'Class', 'y']
    for col in possible_labels:
        if col in df.columns:
            label_column = col
            break
    else:
        raise ValueError(f"❌ No label column found. Please specify one of: {df.columns}")

# -------------------------------
# 3. 分离特征和标签
# -------------------------------
# 特征列：所有以 'embedding_dim_' 开头的列
feature_columns = [col for col in df.columns if col.startswith("embedding_dim_")]

if len(feature_columns) == 0:
    raise ValueError("❌ No embedding feature columns found! Check column name prefix.")

X = df[feature_columns].values        # 特征矩阵 (N, 1024)
y = df[label_column].values           # 标签向量 (N,)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique classes: {np.unique(y)}")

# -------------------------------
# 4. 划分训练集和测试集
# -------------------------------
test_size = 0.1
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y  # 保持类别比例
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# -------------------------------
# 5. 训练随机森林模型
# -------------------------------
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=500,      # 决策树数量
    max_depth=None,        # 不限制深度
    min_samples_split=5,   # 内部节点再划分所需最小样本数
    min_samples_leaf=2,    # 叶子节点最少样本数
    n_jobs=-1,             # 使用所有 CPU 核心
    random_state=random_state,
    verbose=0              # 不输出训练过程
)

# 训练模型
rf_model.fit(X_train, y_train)

# -------------------------------
# 6. 预测与评估
# -------------------------------
y_pred = rf_model.predict(X_test)

# 准确率
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc:.4f}")

# 分类报告（精确率、召回率、F1）
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. 混淆矩阵可视化
# -------------------------------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# -------------------------------
# 8. （可选）特征重要性可视化（前20个最重要维度）
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
# 9. （可选）保存模型
# -------------------------------
import joblib

model_save_path = "../models/rf_protein_classifier.pkl"
joblib.dump(rf_model, model_save_path)
print(f"\n💾 Model saved to '{model_save_path}'")