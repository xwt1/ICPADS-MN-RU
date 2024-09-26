import faiss
import numpy as np
import time
import matplotlib.pyplot as plt

import sys

# 将第一个参数赋值给root_path
root_path = sys.argv[1]

# 文件路径
base_path = root_path + '/data/sift/sift_base.fvecs'
query_path = root_path + '/data/sift/sift_query.fvecs'
ground_truth_path = root_path + '/data/sift/sift_groundtruth.ivecs'

# 读取 fvecs 格式的文件
def read_fvecs(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype='int32')
        dim = data[0]  # 第一个数是维度
        f.seek(0)  # 返回到文件的开头
        data = np.fromfile(f, dtype='float32')
        return data.reshape(-1, dim + 1)[:, 1:]  # 每个向量前有维度信息，所以要跳过

# 读取 ivecs 格式的文件
def read_ivecs(filename):
    data = np.fromfile(filename, dtype='int32')
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:]

# 加载数据
base_data = read_fvecs(base_path)
query_data = read_fvecs(query_path)
ground_truth = read_ivecs(ground_truth_path)

# 构建Faiss索引
d = base_data.shape[1]  # 数据的维度
nlist = 50  # IVF索引的簇数
quantizer = faiss.IndexFlatL2(d)  # 用于IVF的量化器
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 训练索引
index.train(base_data)

# 添加数据
index.add(base_data)
print(f"Initial number of vectors: {index.ntotal}")

# 随机删除部分数据
np.random.seed(42)
n_remove = 10  # 随机删除的向量数量
remove_ids = np.random.choice(index.ntotal, n_remove, replace=False)
index.remove_ids(faiss.IDSelectorBatch(remove_ids))
print(index.ntotal)

# 回填删除的向量，记录每次回填的时间
free_ids = list(remove_ids)  # 记录空闲ID
print(free_ids)
# update_times = []
sum_update_times = 0

for i in range(n_remove):
    start_time = time.time()
    id_to_add = free_ids.pop(0)
    index.add_with_ids(base_data[id_to_add:id_to_add+1], np.array([id_to_add], dtype=np.int64))
    total_query_time = time.time() - start_time
    sum_update_times += total_query_time
    # update_times.append(time.time() - start_time)
    print(index.ntotal)

# 计算空位回填的平均时间
avg_update_time = sum_update_times / (n_remove)
print(f"Average replace_update time: {avg_update_time:.6f} seconds")

num_threads = 40  # 例如，使用 4 个线程
faiss.omp_set_num_threads(num_threads)
print(faiss.omp_get_max_threads())

# 测试不同的nprobe来测量recall和查询速度
recall_values = []
time_values = []
k = 100  # 查询时要返回的最近邻向量数
start_nprobe = 25  # 从最大的 nlist 开始，确保 recall 较高

# 设置每个 nprobe 进行多次查询以取平均值
num_iterations = 1  # 每个 nprobe 进行的查询次数
batch_size = len(query_data)  # 批量查询大小，以更好地利用多线程

print(f"{'Recall':<10} {'Avg Query Time (s)':<20}")

# 执行一次完整查询后，计算平均速度和总的recall
for nprobe in range(start_nprobe, 0, -1):  # 从较高的 nprobe 值开始逐渐减小
    index.nprobe = nprobe

    # 开始计时，执行批量查询
    start_time = time.time()
    D, I = index.search(query_data, k)  # 一次查询整个 query_data
    total_query_time = time.time() - start_time  # 批量查询总时间

    # 计算批量查询的 recall
    total_recall = 0
    for i in range(len(query_data)):
        correct = np.isin(I[i, :], ground_truth[i, :k]).sum()
        total_recall += correct / k  # 累积recall

    # 计算平均查询时间
    avg_query_time = total_query_time / len(query_data)  # 平均每个查询的时间
    time_values.append(avg_query_time)

    # 计算平均 recall
    avg_recall = total_recall / len(query_data)  # 总的 recall
    recall_values.append(avg_recall)

    # 输出平均 recall 和平均查询时间
    # if avg_recall >= 0.9:  # 从 recall 90% 开始显示
    print(f"{avg_recall:.4f}     {avg_query_time:.6f}")

# # 测试不同的时间来测量recall
# recall_values = []
# time_values = []
# k = 100  # 查询时要返回的最近邻向量数
# for nprobe in range(1, nlist+1, 10):
#     index.nprobe = nprobe
#     start_time = time.time()
#     D, I = index.search(query_data, k)
#     query_time = time.time() - start_time
#     time_values.append(query_time)
#
#     # 计算recall
#     correct = 0
#     for i in range(len(query_data)):
#         correct += np.isin(I[i, :], ground_truth[i, :k]).sum()
#     recall = correct / (len(query_data) * k)
#     recall_values.append(recall)
#
# # 绘制recall-time图
# plt.plot(time_values, recall_values, marker='o')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Recall')
# plt.title('Recall-Time Graph')
# plt.grid()
# plt.show()
