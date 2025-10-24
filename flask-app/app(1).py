from flask import Flask, request, jsonify, render_template
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# -------------------------------
# 加载模型和 scaler
# -------------------------------
model_path = "models/dnn_protein_classifier.pth"
scaler_path = "models/feature_scaler.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型定义（与训练时一致）
class ProteinDNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, num_classes))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 加载模型检查点
checkpoint = torch.load(model_path, map_location=device)
input_dim = checkpoint['input_dim']
num_classes = checkpoint['num_classes']
hidden_dims = checkpoint['hidden_dims']
expected_feature_columns = sorted(checkpoint['feature_columns'])  # 训练时的特征列（排序）

model = ProteinDNN(input_dim, num_classes, hidden_dims)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 加载标准化器
scaler = joblib.load(scaler_path)

print("✅ Model and scaler loaded successfully.")


# -------------------------------
# 路由：主页
# -------------------------------
@app.route('/')
def home():
    # 将特征列传递给前端（用于提示）
    feature_cols = list(expected_feature_columns)
    return render_template('index.html', feature_columns=feature_cols)


# -------------------------------
# 路由：上传 Parquet 文件进行预测
# -------------------------------
@app.route('/predict_parquet', methods=['POST'])
def predict_parquet():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.parquet'):
        return jsonify({'error': 'Only .parquet files are allowed'}), 400

    try:
        df = pd.read_parquet(file)
        print("📁 Columns in uploaded file:", df.columns.tolist())

        if 'label' not in df.columns:
            return jsonify({'error': "❌ Column 'label' not found in the Parquet file."}), 400

        # ✅ 确保只使用训练时的特征列（1024 列）
        required_features = set(expected_feature_columns)
        available_features = set(df.columns) - {'label'}

        missing = required_features - available_features
        if missing:
            return jsonify({'error': f"缺少必要特征列: {list(missing)}"}), 400

        extra = available_features - required_features
        if extra:
            print(f"⚠️ 忽略额外列: {list(extra)}")

        X = df[expected_feature_columns].values  # 严格取 1024 列
        y_true = df['label'].values

        if X.shape[1] != input_dim:
            return jsonify({
                'error': f'Feature dimension mismatch after selection: expected {input_dim}, got {X.shape[1]}'
            }), 400

        # 标准化 & 预测
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        with torch.no_grad():
            outputs = model(X_tensor)
            _, preds = torch.max(outputs, 1)
            y_pred = preds.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        # 绘图
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(np.unique(y_true)),
                    yticklabels=sorted(np.unique(y_true)))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        results = df[expected_feature_columns].copy()
        results['true_label'] = y_true
        results['predicted_label'] = y_pred

        return jsonify({
            'accuracy': float(acc),
            'classification_report': report,
            'confusion_matrix_image': img_base64,
            'total_samples': len(df),
            'predictions': results.head(50).to_dict(orient='records')
        })

    except Exception as e:
        # ✅ 关键：必须在这里 return！否则函数返回 None
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

    # ✅ 不要让函数自然结束！上面必须 return


if __name__ == '__main__':
    app.run(debug=True)