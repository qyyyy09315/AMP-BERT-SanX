import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# ----------------------------
# 1. 设置路径
# ----------------------------
model_path = r"D:\PyProject\25卓越杯大数据\prot_bert_bfd"
csv_file = r"D:\PyProject\25卓越杯大数据\Data\test_re.csv"
output_csv_with_vectors = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_vectors.csv"

# ----------------------------
# 2. 加载模型和分词器
# ----------------------------
print("Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
model = BertModel.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# ----------------------------
# ----------------------------
# 3. 读取 CSV 文件并删除 label == 0.0 的行
# ----------------------------
print("Reading CSV file...")
df = pd.read_csv(csv_file)

print(f"原始数据列名: {df.columns.tolist()}")
print(f"原始数据形状: {df.shape}")

# 查找标签列（支持 label, Label, class 等）
label_column = None
for col in df.columns:
    if col.lower() in ['label', 'class', 'target']:
        label_column = col
        break

if label_column is None:
    raise ValueError("未找到标签列（如 'label', 'Label', 'class' 等）")

# 删除 label == 0.0 的所有行
print(f"删除标签为 0.0 的行...")
initial_len = len(df)
df = df[df[label_column] != 0.0].reset_index(drop=True)
print(f"已删除 {initial_len - len(df)} 行（label == 0.0），剩余 {len(df)} 行")

# 自动检测序列列
sequence_column = None
for col in df.columns:
    if col.lower() == 'sequence' or col == 'Sequence':
        sequence_column = col
        break

if sequence_column is None:
    sequence_column = df.columns[0]
    print(f"未找到明确的序列列，使用第一列: {sequence_column}")
else:
    print(f"使用序列列: {sequence_column}")

sequences = df[sequence_column].tolist()

# ----------------------------
# 4. 定义序列到向量的函数
# ----------------------------
def sequence_to_vector(sequence, tokenizer, model, device):
    sequence = ' '.join(list(sequence))
    inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    vector = cls_embedding.cpu().numpy().flatten()
    return vector

# ----------------------------
# 5. 批量处理所有序列
# ----------------------------
print(f"Processing {len(sequences)} sequences...")
vectors = []

for i, seq in enumerate(sequences):
    if i % 10 == 0:
        print(f"Processing sequence {i + 1}/{len(sequences)}")
    try:
        vec = sequence_to_vector(seq, tokenizer, model, device)
        vectors.append(vec)
    except Exception as e:
        print(f"Error processing sequence {i}: {e}")
        vectors.append(np.zeros(model.config.hidden_size))

# ----------------------------
# 6. 转换为 numpy 数组
# ----------------------------
vectors = np.array(vectors)
print(f"特征向量形状: {vectors.shape}")

# ----------------------------
# ----------------------------
# 7. 保存结果（删除序列列和标签列）
# ----------------------------

# 创建特征向量DataFrame
df_vectors = pd.DataFrame(vectors, columns=[f"feat_{i}" for i in range(vectors.shape[1])])

# 删除序列列和标签列
columns_to_drop = [sequence_column, label_column]
df_without_sequence = df.drop(columns=columns_to_drop).reset_index(drop=True)

# 合并数据（不包含序列列和标签列）
df_combined = pd.concat([df_without_sequence, df_vectors], axis=1)

# 验证结果
print(f"\n合并后的数据形状: {df_combined.shape}")
print(f"合并后的列名: {df_combined.columns.tolist()}")
print(f"原始列数量: {len(df.columns)}")
print(f"删除序列列和标签列后剩余列数量: {len(df_without_sequence.columns)}")
print(f"新增特征列数量: {len(df_vectors.columns)}")
print(f"总列数量: {len(df_combined.columns)}")

# 显示前几行验证
print("\n合并后的数据前3行（不包含序列列和标签列）:")
print(df_combined.head(3))

# 保存结果到新文件
output_csv_deleted = r"D:\PyProject\25卓越杯大数据\data2\test_re_with_delete.csv"
df_combined.to_csv(output_csv_deleted, index=False)
print(f"\n结果已保存到: {output_csv_deleted}")
print("Done!")