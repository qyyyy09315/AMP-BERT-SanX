import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import os

# -------------------------------
# 1. 设置设备与混合精度
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 启用混合精度（AMP）
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# -------------------------------
# 2. 加载模型和分词器
# -------------------------------
model_dir = "../prot_bert_bfd"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertModel.from_pretrained(model_dir).to(device).eval()

# -------------------------------
# 3. 读取数据
# -------------------------------
df = pd.read_csv("../Data/train_cl.csv")
sequence_column = "Sequence"
sequences = df[sequence_column].astype(str).tolist()
print(f"Loaded {len(sequences)} sequences.")

# -------------------------------
# 4. 预处理：一次性添加空格（向量化）
# -------------------------------
print("Preprocessing sequences (adding spaces)...")
sequences_with_space = [" ".join(seq) for seq in sequences]  # list of str

# -------------------------------
# 5. 批量提取嵌入（核心优化）
# -------------------------------
def get_embeddings_batch_optimized(sequences, batch_size=64):
    all_embeddings = []
    total_batches = (len(sequences) - 1) // batch_size + 1

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size),
                      total=total_batches,
                      desc="Processing Batches",
                      unit="batch"):

            batch = sequences[i:i + batch_size]

            # 批量编码，直接返回 tensor
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                # 关键：告诉 tokenizer 输入已经是词列表（氨基酸）
                is_split_into_words=False,  # 因为我们传的是 "A B C D" 这样的字符串
            ).to(device)  # 一行搞定所有 tensor 移动

            # 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(**inputs)

            # 提取 [CLS] 并移回 CPU
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(cls_embeddings)

    # 拼接所有 embedding
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    return embeddings_tensor.numpy()

# -------------------------------
# 6. 设置更大的 batch_size（根据显存调整）
# -------------------------------
BATCH_SIZE = 64  # RTX 3090/4090 可尝试 64~128；显存不足请降至 32/16
print(f"Starting embedding extraction with batch_size={BATCH_SIZE}...")

embeddings_array = get_embeddings_batch_optimized(sequences_with_space, batch_size=BATCH_SIZE)
print(f"Embeddings shape: {embeddings_array.shape}")

# -------------------------------
# 7. 保存到 Parquet
# -------------------------------
try:
    import pyarrow  # 触发导入检查
except ImportError:
    raise ImportError("Please install pyarrow: pip install pyarrow")

emb_dim = embeddings_array.shape[1]
emb_columns = [f"embedding_dim_{i}" for i in range(emb_dim)]
df_embeddings = pd.DataFrame(embeddings_array, columns=emb_columns, index=df.index)

df_final = df.drop(columns=[sequence_column]).copy()
df_final = pd.concat([df_final, df_embeddings], axis=1)

output_path = "train_em_cl.parquet"
df_final.to_parquet(output_path, index=False)
print(f"✅ Saved to '{output_path}'")

print(f"Final dataset shape: {df_final.shape}")