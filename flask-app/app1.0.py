from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer

app = Flask(__name__)

# -------------------------------
# 重新定义DataPreprocessor类
# -------------------------------
class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self):
        self.transformer = None
        self.selected_features = None
        self.feature_names = None
        self.important_features = ['feat_339', 'feat_348', 'feat_338', 'Length', 'feat_342',
                                   'feat_347', 'feat_341', 'feat_340']

    def intelligent_feature_engineering(self, X):
        """智能特征工程"""
        print("进行智能特征工程...")
        X_engineered = X.copy()

        # 为重要特征创建高级变换
        for feat in self.important_features:
            if feat in X.columns:
                # 非线性变换
                X_engineered[f'{feat}_squared'] = X[feat] ** 2
                X_engineered[f'{feat}_cubed'] = X[feat] ** 3
                X_engineered[f'{feat}_log'] = np.log1p(np.abs(X[feat]) + 1e-8)
                X_engineered[f'{feat}_reciprocal'] = 1 / (np.abs(X[feat]) + 1e-8)

        # 重要特征之间的高级交互
        for i in range(min(4, len(self.important_features))):
            for j in range(i + 1, min(6, len(self.important_features))):
                feat1, feat2 = self.important_features[i], self.important_features[j]
                if feat1 in X.columns and feat2 in X.columns:
                    X_engineered[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                    X_engineered[f'{feat1}_div_{feat2}'] = X[feat1] / (X[feat2] + 1e-8)
                    X_engineered[f'{feat1}_plus_{feat2}'] = X[feat1] + X[feat2]
                    X_engineered[f'{feat1}_minus_{feat2}'] = X[feat1] - X[feat2]

        # 高级统计特征
        available_features = [f for f in self.important_features if f in X.columns]
        if available_features:
            X_engineered['top_features_mean'] = X[available_features].mean(axis=1)
            X_engineered['top_features_std'] = X[available_features].std(axis=1)
            X_engineered['top_features_range'] = X[available_features].max(axis=1) - X[available_features].min(axis=1)
            X_engineered['top_features_skew'] = X[available_features].skew(axis=1)

        print(f"智能特征工程后维度: {X_engineered.shape}")
        return X_engineered

    def transform_features(self, X):
        """应用特征选择"""
        if self.selected_features is not None:
            return X[self.selected_features]
        return X

    def transform_data(self, X):
        """应用数据变换"""
        if self.transformer is not None:
            return self.transformer.transform(X)
        return X.values

# -------------------------------
# 加载模型和预处理器
# -------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "ultimate_extra_trees_model.pth")

print(f"📁 当前目录: {current_dir}")
print(f"🔍 模型路径: {model_path}")

# 全局变量
model = None
preprocessor = None

def load_model():
    """加载模型函数"""
    global model, preprocessor
    
    try:
        if os.path.exists(model_path):
            print("🔄 正在加载模型...")
            
            # 加载模型数据
            model_data = joblib.load(model_path)
            print(f"📊 加载的数据类型: {type(model_data)}")
            print(f"📊 加载的数据键: {list(model_data.keys()) if isinstance(model_data, dict) else '不是字典'}")
            
            # 调试：打印模型数据的详细信息
            if isinstance(model_data, dict):
                for key, value in model_data.items():
                    print(f"  {key}: {type(value)}")
            
            # 提取模型和预处理器
            if isinstance(model_data, dict):
                # 如果保存的是字典格式
                if 'best_model' in model_data:
                    model = model_data['best_model']
                elif 'model' in model_data:
                    model = model_data['model']
                else:
                    # 尝试直接使用第一个值作为模型
                    first_key = list(model_data.keys())[0]
                    first_value = model_data[first_key]
                    if hasattr(first_value, 'predict'):
                        model = first_value
                    else:
                        model = model_data
                
                # 提取预处理器
                if 'preprocessor' in model_data:
                    preprocessor = model_data['preprocessor']
                else:
                    preprocessor = DataPreprocessor()
            else:
                # 如果直接保存的是模型
                model = model_data
                preprocessor = DataPreprocessor()
            
            # 验证模型
            if hasattr(model, 'predict'):
                print("✅ 模型加载成功")
                print(f"📊 模型类型: {type(model).__name__}")
                print(f"📊 预处理器类型: {type(preprocessor).__name__}")
            else:
                print("❌ 加载的对象不是有效的模型")
                model = None
                preprocessor = None
                
        else:
            print("❌ 模型文件不存在")
            model = None
            preprocessor = None
            
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        model = None
        preprocessor = None

# 在应用启动时加载模型
load_model()

# -------------------------------
# 路由定义
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify({
        'model_loaded': model is not None and hasattr(model, 'predict'),
        'model_type': type(model).__name__ if model else 'None',
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if model is None or not hasattr(model, 'predict'):
        return jsonify({'error': '模型未正确加载或不是有效的预测模型'}), 500

    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': '只支持CSV文件'}), 400

    try:
        # 读取数据
        df = pd.read_csv(file)
        print("📁 上传文件的列:", df.columns.tolist())

        # 检查必要的列
        if 'value' not in df.columns:
            return jsonify({'error': "CSV文件中缺少 'value' 列"}), 400

        # 分离特征和目标
        feature_columns = [col for col in df.columns if col != 'value']
        X = df[feature_columns]
        y_true = df['value'].values

        print(f"📊 原始数据形状: {X.shape}")

        # 使用预处理器进行特征工程
        X_eng = preprocessor.intelligent_feature_engineering(X)
        print(f"📊 特征工程后形状: {X_eng.shape}")
        
        X_sel = preprocessor.transform_features(X_eng)
        print(f"📊 特征选择后形状: {X_sel.shape}")
        
        X_trans = preprocessor.transform_data(X_sel)
        print(f"📊 数据变换后形状: {X_trans.shape}")

        # 预测
        print("🔮 开始预测...")
        y_pred = model.predict(X_trans)
        print(f"📊 预测完成，预测结果形状: {y_pred.shape}")

        # 计算回归指标
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"📈 模型评估 - R²: {r2:.4f}, RMSE: {rmse:.4f}")

        # 创建预测结果图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'预测结果对比 (R² = {r2:.4f})')
        plt.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # 准备返回结果
        results = df.copy()
        results['predicted_value'] = y_pred
        results['residual'] = y_true - y_pred

        return jsonify({
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'prediction_plot': img_base64,
            'total_samples': len(df),
            'predictions': results.head(20).to_dict(orient='records'),
            'statistics': {
                'true_mean': float(np.mean(y_true)),
                'pred_mean': float(np.mean(y_pred)),
                'true_std': float(np.std(y_true)),
                'pred_std': float(np.std(y_pred))
            }
        })

    except Exception as e:
        print(f"❌ 预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"处理错误: {str(e)}"}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """重新加载模型的路由"""
    global model, preprocessor
    load_model()
    
    if model and hasattr(model, 'predict'):
        return jsonify({'status': 'success', 'message': '模型重新加载成功'})
    else:
        return jsonify({'status': 'error', 'message': '模型重新加载失败'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)