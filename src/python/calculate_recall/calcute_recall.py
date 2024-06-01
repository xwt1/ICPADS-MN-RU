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
data_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_base.fvecs'
query_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_query.fvecs'
data = load_fvecs(data_path)
queries = load_fvecs(query_path)[:100]  # Only take first 1000 queries
k = 5

# Set index paths and offsets
index_ranges = [
    (0, 500000),
    (100000, 600000),
    (200000, 700000),
    (300000, 800000),
    (400000, 900000),
    (500000, 1000000)
]
index_paths = [
    '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/issue_statement_index/delete_update/initial_index',
    '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/issue_statement_index/delete_update/index_iteration_1',
    '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/issue_statement_index/delete_update/index_iteration_2',
    '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/issue_statement_index/delete_update/index_iteration_3',
    '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/issue_statement_index/delete_update/index_iteration_4',
    '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/issue_statement_index/delete_update/index_iteration_5'
]
# ef = 500  # Adjust this value to control the trade-off
ef = 200  # Adjust this value to control the trade-off

# Calculate recall and timing for each index
for (start, end), index_path in zip(index_ranges, index_paths):
    sub_data = data[start:end]
    indices_ground_truth, _ = brute_force_knn(sub_data, queries, k)
    labels, load_time, query_time = query_hnsw(index_path, queries, k, ef)
    recall = recall_score(indices_ground_truth, labels, offset=start)
    print(f"RECALL for index {index_path}:", recall)
    print(f"Load time for index {index_path}:", load_time, "seconds")
    print(f"Query time for index {index_path}:", query_time, "seconds")
