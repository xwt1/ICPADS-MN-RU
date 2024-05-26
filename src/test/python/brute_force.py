import numpy as np

def load_fvecs(filename):
    """读取.fvecs文件格式"""
    with open(filename, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # Dimensionality
        n = int(len(f.read()) / (d + 1) / 4)
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, d + 1)[:, 1:].copy()  # 忽略每行的第一个元素（向量的维度）
        return data

def brute_force_knn(data, queries, k, start_idx, end_idx):
    """执行暴力k近邻搜索
    data: 数据集
    queries: 查询集
    k: 最近邻的数量
    start_idx: 数据集开始的下标
    end_idx: 数据集结束的下标
    """
    # 选定子集
    data_subset = data[start_idx:end_idx+1]

    # 初始化返回结果
    indices = np.zeros((len(queries), k), dtype=int)
    distances = np.zeros((len(queries), k), dtype=float)

    # nums = 4
    # # 对每个查询向量执行搜索
    # for i, query in enumerate(queries):
    #     # 计算L2距离
    #     dists = np.linalg.norm(data_subset - query, axis=1)
    #     # 获取最小的k个距离的下标和距离
    #     nearest_indices = np.argsort(dists)[:k]
    #     indices[i] = nearest_indices + start_idx  # 纠正下标为全局下标
    #     distances[i] = dists[nearest_indices]
    #     if i > nums:
    #         break
    # 对最后三个查询向量执行搜索
    for i, query in enumerate(queries[-3:]):  # 只取最后三个查询
        # 计算L2距离
        dists = np.linalg.norm(data_subset - query, axis=1)
        # 获取最小的k个距离的下标和距离
        nearest_indices = np.argsort(dists)[:k]
        # 调整索引数组的填充位置
        adjusted_index = len(queries) - 3 + i
        indices[adjusted_index] = nearest_indices + start_idx  # 纠正下标为全局下标
        distances[adjusted_index] = dists[nearest_indices]

    return indices, distances

# 加载数据和查询向量
data_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs'
query_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs'
data = load_fvecs(data_path)
queries = load_fvecs(query_path)

# 计算使用后半部分数据
n = len(data)
start_idx = n // 2
end_idx = n - 1
k = 5

# 执行暴力搜索
indices, distances = brute_force_knn(data, queries, k, start_idx, end_idx)
print("Indices:\n", indices)
print("Distances:\n", distances)
