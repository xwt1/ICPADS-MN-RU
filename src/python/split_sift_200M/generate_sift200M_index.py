# import hnswlib
# import numpy as np
# import os
#
#
# def read_bvecs_batch(file_path, total_vectors, batch_size):
#     """分批读取bvecs文件"""
#     with open(file_path, 'rb') as f:
#         dimension = int(np.fromfile(f, dtype='int32', count=1)[0])
#         vector_size = 4 + dimension  # 每个向量的字节数（4字节的维度信息 + 向量数据）
#         f.seek(0)  # 重置文件指针
#         for offset in range(0, total_vectors, batch_size):
#             vectors_to_read = min(batch_size, total_vectors - offset)
#             data = np.fromfile(f, dtype='uint8', count=vector_size * vectors_to_read)
#             data = data.reshape(-1, vector_size)
#             vectors = data[:, 4:].astype('float32')
#             yield vectors
#
#
# # 参数设置
# total_vectors = 100000000  # 前2亿个向量
# batch_size = 1_000_000  # 每次读取100万数据，可根据内存调整
# dimension = 128  # SIFT特征维度
#
# # 初始化HNSW索引
# index = hnswlib.Index(space='l2', dim=dimension)  # 使用L2距离
#
# # 设置索引参数
# index.init_index(max_elements=total_vectors, ef_construction=200, M=16)  # 可根据需要调整ef_construction和M
# index.set_ef(200)  # 查询时的ef参数
#
# data_path = '/root/WorkSpace/dataset/sift/bigann_base.bvecs'
#
# # 创建保存索引的目录（如果不存在）
# save_dir = '/root/WorkSpace/dataset/sift/sift200M/index'
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, 'sift_200M_index.bin')
#
# # 分批读取数据并添加到索引
# vector_count = 0
# for batch_vectors in read_bvecs_batch(data_path, total_vectors, batch_size):
#     num_vectors = batch_vectors.shape[0]
#     index.add_items(batch_vectors)
#     vector_count += num_vectors
#     print(f"已添加 {vector_count} 个向量到索引。")
#     # 可根据需要定期保存索引
#     # index.save_index(save_path)
#
# # 最终保存完整索引
# index.save_index(save_path)
# print(f"索引已保存到 {save_path}")


import hnswlib
import numpy as np
import os
import time  # 导入time模块用于计时

def read_bvecs_batch(file_path, total_vectors, batch_size):
    """分批读取bvecs文件"""
    with open(file_path, 'rb') as f:
        dimension = int(np.fromfile(f, dtype='int32', count=1)[0])
        vector_size = 4 + dimension  # 每个向量的字节数（4字节的维度信息 + 向量数据）
        f.seek(0)  # 重置文件指针
        for offset in range(0, total_vectors, batch_size):
            vectors_to_read = min(batch_size, total_vectors - offset)
            data = np.fromfile(f, dtype='uint8', count=vector_size * vectors_to_read)
            data = data.reshape(-1, vector_size)
            vectors = data[:, 4:].astype('float32')
            yield vectors

# 参数设置
total_vectors = 100_000_000  # 前1亿个向量
batch_size = 1_000_000       # 每次读取100万数据，可根据内存调整
dimension = 128              # SIFT特征维度

# 初始化HNSW索引
index = hnswlib.Index(space='l2', dim=dimension)  # 使用L2距离

# 设置索引参数
index.init_index(max_elements=total_vectors, ef_construction=200, M=16)  # 可根据需要调整ef_construction和M
index.set_ef(200)  # 查询时的ef参数

data_path = '/root/WorkSpace/dataset/sift/bigann_base.bvecs'

# 创建保存索引的目录（如果不存在）
save_dir = '/root/WorkSpace/dataset/sift/sift200M/index'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'sift_200M_index.bin')

# 开始计时
start_time = time.time()

# 分批读取数据并添加到索引
vector_count = 0
for batch_vectors in read_bvecs_batch(data_path, total_vectors, batch_size):
    num_vectors = batch_vectors.shape[0]
    index.add_items(batch_vectors)
    vector_count += num_vectors
    print(f"已添加 {vector_count} 个向量到索引。")
    # 可根据需要定期保存索引
    # index.save_index(save_path)

# 最终保存完整索引
index.save_index(save_path)
print(f"索引已保存到 {save_path}")

# 结束计时并计算总耗时
end_time = time.time()
total_time = end_time - start_time

# 将总耗时转换为小时、分钟、秒
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"构建索引总耗时：{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
