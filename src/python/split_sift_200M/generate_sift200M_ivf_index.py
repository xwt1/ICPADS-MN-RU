import faiss
import numpy as np
import os
import time
import gc  # 导入垃圾回收模块

def read_bvecs_batch(file_path, batch_size, total_vectors):
    """生成器函数，分批读取 bvecs 文件"""
    with open(file_path, 'rb') as f:
        # 读取第一个向量的维度信息
        dim_bytes = f.read(4)
        if not dim_bytes:
            return  # 文件为空
        dim = int(np.frombuffer(dim_bytes, dtype='int32')[0])

        vector_size = 4 + dim  # 每个向量的字节数（4 字节的维度信息 + 向量数据）
        f.seek(0)  # 重置文件指针

        vectors_read = 0
        while vectors_read < total_vectors:
            # 计算本次需要读取的向量数量
            vectors_to_read = min(batch_size, total_vectors - vectors_read)
            # 计算需要读取的总字节数
            bytes_to_read = vectors_to_read * vector_size
            # 读取数据
            data = f.read(bytes_to_read)
            if not data:
                break  # 文件读取完毕
            # 将数据转换为 numpy 数组
            batch = np.frombuffer(data, dtype='uint8')
            batch = batch.reshape(-1, vector_size)
            # 跳过每个向量前面的 4 字节维度信息
            vectors = batch[:, 4:].astype('float32')
            yield vectors
            vectors_read += vectors.shape[0]
            # 显式删除不再需要的变量并进行垃圾回收
            del data
            del batch
            gc.collect()

# 参数设置
total_vectors = 100_000_000  # 前 1 亿个向量
batch_size = 1_000_000       # 每次读取 100 万数据，可根据内存调整
dimension = 128              # SIFT 特征维度

data_path = '/root/WorkSpace/dataset/sift/bigann_base.bvecs'
index_path = '/root/WorkSpace/dataset/sift/sift200M/index/faiss_ivf_index.bin'

# 检查索引是否已存在
# if os.path.exists(index_path):
#     print("索引文件已存在，无需重新构建。")
#     exit()

# 创建保存索引的目录（如果不存在）
save_dir = os.path.dirname(index_path)
os.makedirs(save_dir, exist_ok=True)

# 初始化 Faiss 索引参数
nlist = 1000000  # 聚类中心的数量，可根据数据规模调整
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# **使用部分数据进行训练**
train_size = 10_000_000  # 用于训练的向量数量，可根据内存情况调整
print(f"正在加载 {train_size} 个向量用于训练索引...")

# 从数据中读取用于训练的向量
train_vectors = next(read_bvecs_batch(data_path, train_size, train_size))

print("正在训练索引...")
index.train(train_vectors)
print("索引训练完成。")

# 训练数据用完后，释放内存
del train_vectors
gc.collect()

# 开始计时
start_time = time.time()

# 分批读取数据并添加到索引
vector_count = 0
print("开始分批添加向量到索引中...")
for batch_vectors in read_bvecs_batch(data_path, batch_size, total_vectors):
    num_vectors = batch_vectors.shape[0]
    index.add(batch_vectors)
    vector_count += num_vectors
    print(f"已添加 {vector_count} 个向量到索引。")

    # 处理完一批数据后，释放内存
    del batch_vectors
    gc.collect()

    if vector_count >= total_vectors:
        break  # 已添加指定数量的向量

# 保存索引到磁盘
print("正在保存索引到磁盘...")
faiss.write_index(index, index_path)
print(f"索引已保存到 {index_path}")

# 结束计时并计算总耗时
end_time = time.time()
total_time = end_time - start_time

# 将总耗时转换为小时、分钟、秒
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"构建索引总耗时：{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
