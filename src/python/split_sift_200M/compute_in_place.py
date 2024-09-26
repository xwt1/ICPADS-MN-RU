import hnswlib
import numpy as np
import time
from multiprocessing.dummy import Pool as ThreadPool  # 使用线程池

# 设置参数
dim = 128
num_elements = 1000000  # 删除并添加回的向量数量

index_path = '/root/WorkSpace/dataset/sift/sift200M/index/sift_100M_index.bin'
data_path = '/root/WorkSpace/dataset/sift/bigann_base.bvecs'

# 初始化索引，设置 allow_replace_deleted=True
p = hnswlib.Index(space='l2', dim=dim)
p.load_index(index_path, allow_replace_deleted=True)

# 设置ef参数（可选）
p.set_ef(50)

# 设置多线程数
p.set_num_threads(40)

# 读取前一百万个向量
def read_bvecs(filename, num_vectors):
    with open(filename, 'rb') as f:
        # 读取维度
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        vector_size = 4 + dim

        # 回到文件开头
        f.seek(0)

        # 计算总的读取字节数
        total_bytes = num_vectors * vector_size

        # 读取数据
        data = np.fromfile(f, dtype=np.uint8, count=total_bytes)

        # 重塑数组
        data = data.reshape(num_vectors, vector_size)

        # 提取向量部分并转换为float32类型
        vecs = data[:, 4:].astype(np.float32)

    return vecs

data = read_bvecs(data_path, num_elements)

# 创建标签数组
labels = np.arange(num_elements)

# 删除向量并测量时间（使用多线程）
def delete_labels(label):
    p.mark_deleted(int(label))

start_time_delete = time.time()
pool = ThreadPool(40)  # 创建40个线程的线程池
pool.map(delete_labels, labels)
pool.close()
pool.join()
end_time_delete = time.time()
total_time_delete = end_time_delete - start_time_delete
average_time_delete = total_time_delete / num_elements

# 使用replace_deleted=True添加向量并测量时间
start_time_add = time.time()
p.add_items(data, labels, replace_deleted=True)
end_time_add = time.time()
total_time_add = end_time_add - start_time_add
average_time_add = total_time_add / num_elements

total_time = total_time_delete + total_time_add
average_time = total_time / num_elements

print("删除操作总时间：{:.6f} 秒".format(total_time_delete))
print("添加操作总时间：{:.6f} 秒".format(total_time_add))
print("删除和replaced_update的总时间：{:.6f} 秒".format(total_time))
print("每次操作的平均时间：{:.6f} 秒".format(average_time))
