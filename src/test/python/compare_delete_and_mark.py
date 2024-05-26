import hnswlib
import numpy as np
import time

def load_fvecs(filename):
    """读取.fvecs文件格式"""
    with open(filename, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # Dimensionality
        n = int(len(f.read()) / (d + 1) / 4)
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, d + 1)[:, 1:].copy()  # 忽略每行的第一个元素（向量的维度）
        return data

def query_hnsw(index_path, query_vectors_path, k):
    # 加载索引
    p = hnswlib.Index(space='l2', dim=128)  # 假设维度为128
    p.load_index(index_path)

    # 加载查询向量
    queries = load_fvecs(query_vectors_path)

    # 开始查询并计时
    start_time = time.time()
    labels, distances = p.knn_query(queries, k=k)
    total_time = time.time() - start_time

    # 打印查询结果和时间
    print("Query time:", total_time, "seconds")
    return labels, distances, total_time

# 使用函数示例
mark_index_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base_deleted_half.bin'
delete_index_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base.bin'
query_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs'
k = 5

mark_labels ,mark_distances, mark_index_time = query_hnsw(mark_index_path, query_path, k)
delete_lables ,delete_distances, delete_index_time = query_hnsw(delete_index_path, query_path, k)


print("mark_index_time Query time:", mark_index_time, "seconds")
print("delete_index_time Query time:", delete_index_time, "seconds")
print("mark_labels: ",mark_labels)
print("delete_lables: ",delete_lables)
print("mark_distances: ",mark_distances)
print("delete_distances: ",delete_distances)