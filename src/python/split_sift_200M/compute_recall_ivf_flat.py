import faiss
import numpy as np
import time
import os
import csv

def read_bvecs(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    dim = int(np.frombuffer(data[:4], dtype=np.int32)[0])
    return data.reshape(-1, dim + 4)[:, 4:].astype(np.float32)

def read_ivecs(filename):
    data = np.fromfile(filename, dtype=np.int32)
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:]

# 数据路径
index_path = '/root/WorkSpace/dataset/sift/sift200M/index/faiss_ivf_index.bin'
query_path = '/root/WorkSpace/dataset/sift/sift200M/bigann_query.bvecs'
gt_path = '/root/WorkSpace/dataset/sift/sift200M/gnd/idx_100M.ivecs'
output_csv_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/random/faiss_results.csv'

# 加载查询数据
print("正在加载查询数据...")
queries = read_bvecs(query_path)
print(f"查询数据形状: {queries.shape}")

# 加载真值数据
print("正在加载真值数据...")
ground_truth = read_ivecs(gt_path)
print(f"真值数据形状: {ground_truth.shape}")

# 检查索引文件是否存在
if not os.path.exists(index_path):
    print("索引文件不存在，请先运行索引构建程序。")
    exit()

# 加载已构建的索引
print("正在加载索引...")
index = faiss.read_index(index_path)

# 设置查询线程数
num_threads = 40
faiss.omp_set_num_threads(num_threads)
print(f"使用 {num_threads} 个线程进行查询")

# 定义 nprobe 参数值，可以根据需要调整范围和步长
nprobe_values = list(range(100, 1001, 100))

# 准备保存结果的列表
results = []

# 设置 k 值
k = ground_truth.shape[1]

# 对每个 nprobe 值进行搜索，计算召回率和平均查询时间
for nprobe in nprobe_values:
    index.nprobe = nprobe
    print(f"当前设置的 nprobe 值为: {nprobe}")
    start_time = time.time()
    D, I = index.search(queries, k)
    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / len(queries)

    # 计算召回率
    correct = 0
    total = 0
    for i in range(len(queries)):
        gt_set = set(ground_truth[i])
        retrieved_set = set(I[i][:k])
        correct += len(gt_set & retrieved_set)
        total += len(gt_set)
    average_recall = correct / total

    print(f"nprobe={nprobe}, Recall@{k}={average_recall:.4f}, 平均查询时间={average_time:.6f} 秒")

    # 将结果添加到列表
    results.append([nprobe, average_recall, average_time])

# 保存结果到 CSV 文件
output_dir = os.path.dirname(output_csv_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 写入表头
    writer.writerow(['nprobe', 'recall', 'time'])
    # 写入数据
    writer.writerows(results)

print(f"结果已保存到 {output_csv_path}")
