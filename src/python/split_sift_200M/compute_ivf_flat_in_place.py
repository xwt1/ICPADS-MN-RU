import faiss
import numpy as np
import time

# 设置多线程数
faiss.omp_set_num_threads(40)

# 设置参数
dim = 128
num_elements = 1000000  # 删除并添加回的向量数量

index_path = '/root/WorkSpace/dataset/sift/sift200M/index/faiss_ivf_index.bin'
data_path = '/root/WorkSpace/dataset/sift/bigann_base.bvecs'

# 加载索引
index = faiss.read_index(index_path)

# 读取前一百万个向量
def read_bvecs(filename, num_vectors):
    with open(filename, 'rb') as f:
        # 读取维度
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        # 每个向量的字节数：4（维度） + dim（向量数据）
        num_bytes_per_vec = 4 + dim
        total_bytes = num_vectors * num_bytes_per_vec
        data = np.fromfile(f, dtype=np.uint8, count=total_bytes)
        data = data.reshape(-1, num_bytes_per_vec)
        vecs = data[:, 4:].astype('float32')
    return vecs

data = read_bvecs(data_path, num_elements)

# 创建标签数组，确保是 int64 类型并连续存储
labels = np.arange(num_elements).astype(np.int64)
labels = np.ascontiguousarray(labels)

# 删除向量并测量时间
start_time_delete = time.time()

# 使用 faiss.swig_ptr 将 NumPy 数组转换为指针
id_selector = faiss.IDSelectorBatch(num_elements, faiss.swig_ptr(labels))
index.remove_ids(id_selector)

end_time_delete = time.time()
total_time_delete = end_time_delete - start_time_delete
average_time_delete = total_time_delete / num_elements

# 添加向量并测量时间
start_time_add = time.time()
index.add_with_ids(data, labels)
end_time_add = time.time()
total_time_add = end_time_add - start_time_add
average_time_add = total_time_add / num_elements

total_time = total_time_delete + total_time_add
average_time = total_time / num_elements

print("删除操作总时间：{:.6f} 秒".format(total_time_delete))
print("添加操作总时间：{:.6f} 秒".format(total_time_add))
print("删除和添加的总时间：{:.6f} 秒".format(total_time))
print("每次操作的平均时间：{:.6f} 秒".format(average_time))
