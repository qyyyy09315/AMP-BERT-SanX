import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pyarrow as pa
import pyarrow.parquet as pq

# 1. 设置设备：优先使用 GPU (CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 本地模型路径
model_dir = "./prot_bert_bfd"

# 2. 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertModel.from_pretrained(model_dir)
model.to(device)
model.eval()


# 3. 读取序列文件 - 保留所有列
def read_sequences_with_metadata(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    print(f"文件列名: {list(df.columns)}")
    print(f"文件形状: {df.shape}")

    # 读取序列列（第一列）
    sequences = df.iloc[:, 0].dropna().tolist()
    print(f"读取序列列: '{df.columns[0]}'")
    print(f"前3个序列预览: {sequences[:3]}")

    # 保留其他列（Length和label）
    metadata_columns = df.columns[1:].tolist()  # 从第二列开始的所有列
    metadata_df = df[metadata_columns].copy()

    print(f"保留的元数据列: {metadata_columns}")

    return sequences, metadata_df


# 4. 获取嵌入表示
def get_protein_embedding(sequence):
    # 确保序列是字符串类型
    if isinstance(sequence, (int, float)):
        sequence = str(int(sequence))
    elif not isinstance(sequence, str):
        sequence = str(sequence)

    # 去除可能的空格
    sequence = sequence.strip()

    # ProtBert 要求氨基酸之间有空格
    sequence = " ".join(list(sequence))

    # Tokenize 并直接移动到 GPU
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    # 将输入张量移动到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 取 [CLS] token 的 embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)

    return cls_embedding.cpu()


# 5. 保存为Parquet格式的函数
def save_as_parquet(df, file_path):
    """
    将DataFrame保存为Parquet格式
    """
    try:
        # 转换DataFrame为PyArrow Table
        table = pa.Table.from_pandas(df)

        # 保存为Parquet文件
        pq.write_table(table, file_path, compression='snappy')

        print(f"✅ Parquet文件保存成功: {file_path}")
        print(f"文件大小: {pd.DataFrame([file_path]).memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB (估计)")

        # 验证文件
        verify_table = pq.read_table(file_path)
        print(f"验证读取: {verify_table.num_rows} 行, {verify_table.num_columns} 列")

    except Exception as e:
        print(f"❌ 保存Parquet文件时出错: {e}")
        raise


# 主程序
if __name__ == "__main__":
    # 读取序列和元数据
    sequences, metadata_df = read_sequences_with_metadata("Data/train_cl.csv")

    embeddings = []
    valid_sequences = []
    valid_indices = []  # 记录有效的索引

    print(f"Processing {len(sequences)} sequences...")

    for i, seq in enumerate(sequences):
        # 检查序列是否有效
        if pd.isna(seq) or seq == "":
            continue

        # 确保序列是字符串
        if not isinstance(seq, str):
            seq = str(seq)

        # 检查序列长度
        if len(seq) == 0:
            continue

        try:
            emb = get_protein_embedding(seq)
            embeddings.append(emb)
            valid_sequences.append(seq)
            valid_indices.append(i)  # 记录有效序列的原始索引

            if (i + 1) % 100 == 0:  # 每100个序列打印一次进度
                print(f"[{i + 1}/{len(sequences)}] Sequence: {seq[:10]}... -> Embedding shape: {emb.shape}")

        except Exception as e:
            print(f"[{i + 1}/{len(sequences)}] 处理序列时出错: {e}")
            continue

    # 所有 embedding 处理完毕
    print(f"✅ Done! Got {len(embeddings)} embeddings.")

    # 保存嵌入向量
    torch.save(embeddings, "DataAfterBert/train_em_cl.pt")
    print("💾 Embeddings saved to 'train_em_cl.pt'")

    # 保存有效的序列
    with open("DataAfterBert/valid_sequences.txt", "w") as f:
        for seq in valid_sequences:
            f.write(seq + "\n")
    print("💾 Valid sequences saved to 'valid_sequences.txt'")

    # 保存包含元数据的完整结果
    print("\n保存包含元数据的完整结果...")

    # 只保留有效序列对应的元数据
    valid_metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)

    # 创建完整的结果DataFrame（不包含sequence列）
    result_df = pd.DataFrame()

    # 添加元数据列
    for col in valid_metadata_df.columns:
        result_df[col] = valid_metadata_df[col].values

    # 添加嵌入向量信息
    embeddings_array = torch.cat(embeddings, dim=0).numpy()
    result_df['embedding_vector'] = list(embeddings_array)

    # 保存为CSV格式
    result_df.to_csv("DataAfterBert/train_data_complete.csv", index=False)
    print("💾 完整数据保存到 'train_data_complete.csv' (不包含sequence列)")

    # 保存为Parquet格式
    print("\n正在保存为Parquet格式...")
    parquet_file_path = "DataAfterBert/train_data_complete.parquet"
    save_as_parquet(result_df, parquet_file_path)

    # 比较文件大小
    print("\n📊 文件大小比较:")
    try:
        import os

        csv_size = os.path.getsize("DataAfterBert/train_data_complete.csv") / 1024 / 1024
        parquet_size = os.path.getsize(parquet_file_path) / 1024 / 1024

        print(f"CSV文件大小: {csv_size:.2f} MB")
        print(f"Parquet文件大小: {parquet_size:.2f} MB")
        print(f"压缩比: {csv_size / parquet_size:.2f}x")

    except Exception as e:
        print(f"无法比较文件大小: {e}")

    # 显示结果统计
    print(f"\n结果统计:")
    print(f"有效序列数量: {len(valid_sequences)}")
    print(f"保留的元数据列: {list(valid_metadata_df.columns)}")
    print(f"完整数据形状: {result_df.shape}")
    print(f"嵌入向量维度: {embeddings_array.shape[1]}")

    # 显示前几行结果
    print(f"\n前3行完整数据:")
    print(result_df.head(3))

    # 显示内存使用情况
    print(f"\n内存使用统计:")
    memory_usage = result_df.memory_usage(deep=True)
    print(f"总内存使用: {memory_usage.sum() / 1024 / 1024:.2f} MB")
    for col, usage in memory_usage.items():
        print(f"  {col}: {usage / 1024 / 1024:.2f} MB")

    print("\n🎉 所有文件保存完成!")
    print("可用文件格式:")
    print("  - CSV: DataAfterBert/train_data_complete.csv")
    print("  - Parquet: DataAfterBert/train_data_complete.parquet")
