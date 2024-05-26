import time
import hnswlib
import numpy as np

def load_fvecs(filename):
    """Read .fvecs file format"""
    with open(filename, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # Dimensionality
        n = int(len(f.read()) / (d + 1) / 4)
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, d + 1)[:, 1:].copy()  # Ignore the first element of each row (dimension of the vector)
        return data

def brute_force_knn(data, queries, k):
    """Perform brute-force k-NN search, return indices and distances for all queries"""
    indices = np.zeros((len(queries), k), dtype=int)
    distances = np.zeros((len(queries), k), dtype=float)
    for i, query in enumerate(queries):
        dists = np.linalg.norm(data - query, axis=1)
        nearest_indices = np.argsort(dists)[:k]
        indices[i] = nearest_indices
        distances[i] = dists[nearest_indices]
    return indices, distances

def recall_score(ground_truth, predictions, offset=0):
    """Calculate RECALL score, can adjust offset to handle different index start positions"""
    hit_count = 0
    for true_data, pred_data in zip(ground_truth, predictions):
        adjusted_pred_data = [label - offset for label in pred_data if label >= offset]
        hit_count += len(set(true_data).intersection(set(adjusted_pred_data)))
    return hit_count / (len(ground_truth) * ground_truth.shape[1])

def query_hnsw(index_path, queries, k, ef=None):
    # Load index
    start_time = time.time()
    p = hnswlib.Index(space='l2', dim=128)
    p.load_index(index_path)
    load_time = time.time() - start_time

    # Set ef parameter if provided
    if ef is not None:
        p.set_ef(ef)

    # Perform query
    query_start_time = time.time()
    labels, distances = p.knn_query(queries, k=k)
    query_time = time.time() - query_start_time

    return labels, load_time, query_time

# Load data and queries
data_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs'
query_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs'
data = load_fvecs(data_path)
queries = load_fvecs(query_path)[:1000]  # Only take first 1000 queries
k = 10000

# Compute ground truth
indices_ground_truth, _ = brute_force_knn(data[len(data)//2:], queries, k)

# Set index paths and ef parameter
mark_index_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base_deleted_half.bin'
delete_index_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base.bin'
ef = 500  # Adjust this value to control the trade-off

# Calculate recall and timing for marked index
mark_labels, mark_load_time, mark_query_time = query_hnsw(mark_index_path, queries, k, ef)
mark_recall = recall_score(indices_ground_truth, mark_labels, offset=len(data)//2)

# Calculate recall and timing for direct index
delete_labels, delete_load_time, delete_query_time = query_hnsw(delete_index_path, queries, k, ef)
delete_recall = recall_score(indices_ground_truth, delete_labels)

print("RECALL for marked index:", mark_recall)
print("Load time for marked index:", mark_load_time, "seconds")
print("Query time for marked index:", mark_query_time, "seconds")
print("RECALL for direct index on last 500k:", delete_recall)
print("Load time for direct index:", delete_load_time, "seconds")
print("Query time for direct index:", delete_query_time, "seconds")

# import time
# import hnswlib
# import numpy as np
#
# def load_fvecs(filename):
#     """读取.fvecs文件格式"""
#     with open(filename, 'rb') as f:
#         d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # Dimensionality
#         n = int(len(f.read()) / (d + 1) / 4)
#         f.seek(0)
#         data = np.fromfile(f, dtype=np.float32)
#         data = data.reshape(-1, d + 1)[:, 1:].copy()  # 忽略每行的第一个元素（向量的维度）
#         return data
#
# def brute_force_knn(data, queries, k):
#     """执行暴力k近邻搜索，返回所有查询的索引和距离"""
#     indices = np.zeros((len(queries), k), dtype=int)
#     distances = np.zeros((len(queries), k), dtype=float)
#     for i, query in enumerate(queries):
#         dists = np.linalg.norm(data - query, axis=1)
#         nearest_indices = np.argsort(dists)[:k]
#         indices[i] = nearest_indices
#         distances[i] = dists[nearest_indices]
#     return indices, distances
#
# def recall_score(ground_truth, predictions, offset=0):
#     """计算RECALL分数，可调整offset处理不同的索引起始位置"""
#     hit_count = 0
#     for true_data, pred_data in zip(ground_truth, predictions):
#         adjusted_pred_data = [label - offset for label in pred_data if label >= offset]
#         hit_count += len(set(true_data).intersection(set(adjusted_pred_data)))
#     return hit_count / (len(ground_truth) * ground_truth.shape[1])
#
# def query_hnsw(index_path, queries, k, ef=None):
#     p = hnswlib.Index(space='l2', dim=128)
#     p.load_index(index_path)
#     if ef != None:
#         p.set_ef(ef)  # 设置 ef 参数
#     labels, distances = p.knn_query(queries, k=k)
#     return labels
#
# # 加载数据和查询
# data_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs'
# query_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs'
# data = load_fvecs(data_path)
# queries = load_fvecs(query_path)[:2000]  # 只取前1000个查询
# k = 1000
#
# # 计算ground truth
# indices_ground_truth, _ = brute_force_knn(data[len(data)//2:], queries, k)
#
# # 设定索引路径和ef参数
# mark_index_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base_deleted_half.bin'
# delete_index_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base.bin'
# ef = 500  # 这个值可以调整以控制trade-off
#
# # 计算 mark_recall 的时间
# start_time = time.time()
# mark_labels = query_hnsw(mark_index_path, queries, k,ef)
# mark_recall = recall_score(indices_ground_truth, mark_labels, offset=len(data)//2)
# mark_time = time.time() - start_time
#
# # 计算 delete_recall 的时间
# start_time = time.time()
# delete_labels = query_hnsw(delete_index_path, queries, k, ef)
# delete_recall = recall_score(indices_ground_truth, delete_labels)
# delete_time = time.time() - start_time
#
# print("RECALL for marked index:", mark_recall)
# print("Time taken for marked index:", mark_time, "seconds")
# print("RECALL for direct index on last 500k:", delete_recall)
# print("Time taken for direct index:", delete_time, "seconds")
