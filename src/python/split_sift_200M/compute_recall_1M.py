import hnswlib
import numpy as np
import time
import os

def read_bvecs(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    dim = int(np.frombuffer(data[:4], dtype=np.int32)[0])
    return data.reshape(-1, dim + 4)[:, 4:].astype(np.float32)

def read_ivecs(filename):
    data = np.fromfile(filename, dtype=np.int32)
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:]

# 数据路径
index_path = '/root/WorkSpace/dataset/sift/sift200M/index/sift_1M_index.bin'
query_path = '/root/WorkSpace/dataset/sift/sift200M/bigann_query.bvecs'
gt_path = '/root/WorkSpace/dataset/sift/sift200M/gnd/idx_1M.ivecs'

# 加载查询数据
print("正在加载查询数据...")
queries = read_bvecs(query_path)
print(f"查询数据形状: {queries.shape}")

# 加载真值数据
print("正在加载真值数据...")
ground_truth = read_ivecs(gt_path)
print(f"真值数据形状: {ground_truth.shape}")

# 初始化 hnswlib 索引
dim = queries.shape[1]
print("正在初始化索引...")
index = hnswlib.Index(space='l2', dim=dim)

# 加载已构建的索引
print("正在加载索引...")
index.load_index(index_path)

# 获取可用的 CPU 核心数
num_threads = os.cpu_count()
print(f"使用 {num_threads} 个线程进行查询")

# 定义不同的搜索配置（ef 参数值）
ef_values = [10, 50, 100, 2000, 5000]

# 对每个 ef 值进行搜索，计算召回率和时间
for ef in ef_values:
    index.set_ef(ef)
    print(f"当前设置的 ef 值为: {ef}")
    start_time = time.time()
    labels, distances = index.knn_query(queries, k=ground_truth.shape[1], num_threads=num_threads)
    end_time = time.time()
    time_taken = end_time - start_time

    # 计算召回率
    k = ground_truth.shape[1]
    recalls = []
    for i in range(len(queries)):
        gt_set = set(ground_truth[i])
        retrieved_set = set(labels[i][:k])
        recall_i = len(gt_set & retrieved_set) / k
        recalls.append(recall_i)
    average_recall = np.mean(recalls)

    print(f"ef={ef}, Recall@{k}={average_recall:.4f}, 时间={time_taken:.4f} 秒")
