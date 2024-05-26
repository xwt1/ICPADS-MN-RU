import hnswlib
import numpy as np

# 加载SIFT查询数据集
def load_fvecs(file_path):
    with open(file_path, 'rb') as f:
        data = []
        while True:
            # 每个向量的第一个int值表示维度
            dim_array = np.fromfile(f, dtype=np.int32, count=1)
            if dim_array.size == 0:
                break
            dim = dim_array[0]
            vec = np.fromfile(f, dtype=np.float32, count=dim)
            if vec.size != dim:
                break
            data.append(vec)
    return np.vstack(data)

# 加载HNSW索引
def load_hnsw_index(index_path, dim, max_elements, ef=1000000):
    p = hnswlib.Index(space='l2', dim=dim)  # 使用L2距离度量
    p.load_index(index_path, max_elements=max_elements)
    p.set_ef(ef)  # 设置 ef 参数
    return p

# 进行查询
def query_index(index, queries, k=10):
    labels, distances = index.knn_query(queries, k=k)
    return labels, distances

# 文件路径
sift_query_file = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs'
hnsw_index_file = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_delete_over.bin'

# 加载查询数据
queries = load_fvecs(sift_query_file)

# HNSW索引参数
dim = queries.shape[1]  # 数据的维度
max_elements = 1000000  # HNSW索引中的最大元素数

# 加载HNSW索引
index = load_hnsw_index(hnsw_index_file, dim, max_elements)

# 查询HNSW索引
k = 1000000  # 每个查询返回的最近邻数
for i in range(100):
    labels, distances = query_index(index, [queries[i]], k=k)
    print(len(labels[0]))

# 打印查询结果
# print(len(labels))
# for i in range(100):
#     print(len(queries[i]))

# for i in range(len(queries)):
#     print(f"Query {i}:")
#     print("Labels:", labels[i])
#     print("Distances:", distances[i])
