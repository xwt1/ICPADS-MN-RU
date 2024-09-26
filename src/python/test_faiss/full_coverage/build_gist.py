import faiss
import numpy as np
import time
import matplotlib.pyplot as plt

# 文件路径
base_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/gist/gist_base.fvecs'
index_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/gist/faiss_index/gist_index_python.bin'

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
    data = np.fromfile(filename, 'int32')
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:]

# 加载数据
base_data = read_fvecs(base_path)

# 构建Faiss索引
d = base_data.shape[1]  # 数据的维度
nlist = 1000  # IVF索引的簇数
quantizer = faiss.IndexFlatL2(d)  # 用于IVF的量化器
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 训练索引
index.train(base_data)

# 计算添加数据的时间
start_time = time.time()

# 添加数据
index.add(base_data)

end_time = time.time()

# 计算时间差
add_time = end_time - start_time

print(f"Initial number of vectors: {index.ntotal}")
print(f"Time taken to add data: {add_time:.4f} seconds")

# 保存索引到文件
faiss.write_index(index, index_path)
print(f"Index saved to {index_path}")
