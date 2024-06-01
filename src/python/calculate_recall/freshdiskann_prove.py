import time
import hnswlib
import numpy as np
import random

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

def recall_score(ground_truth, predictions):
    """Calculate RECALL score"""
    hit_count = 0
    for true_data, pred_data in zip(ground_truth, predictions):
        hit_count += len(set(true_data).intersection(set(pred_data)))
    return hit_count / (len(ground_truth) * ground_truth.shape[1])

def query_hnsw(index, queries, k):
    # Perform query
    query_start_time = time.time()
    labels, distances = index.knn_query(queries, k=k)
    query_time = time.time() - query_start_time

    return labels, query_time

# Load data and queries
data_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_base.fvecs'
query_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_query.fvecs'

data = load_fvecs(data_path)
queries = load_fvecs(query_path)[:100]  # Only take first 100 queries
k = 5

# Initialize the HNSW index
p = hnswlib.Index(space='l2', dim=data.shape[1])
p.init_index(max_elements=len(data), ef_construction=200, M=16, allow_replace_deleted=True)

# Add data points to the index
p.add_items(data)

p.save_index("/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/freshdiskann_prove/diskann_prove.bin")
# p.set_ef(50)  # ef parameter for querying
#
# # Perform initial brute-force k-NN search to get ground truth
# indices_ground_truth, _ = brute_force_knn(data, queries, k)
#
# # Number of iterations for delete and re-add process
# num_iterations = 10
#
# for iteration in range(num_iterations):
#     # Randomly select 5% of the data to delete
#     num_delete = len(data) // 20
#     delete_indices = random.sample(range(len(data)), num_delete)
#
#     # Save the vectors and their labels to be deleted before deleting them
#     deleted_vectors = data[delete_indices]
#
#     # Delete selected vectors from the index
#     for index in delete_indices:
#         p.mark_deleted(index)
#
#     # Re-add the deleted vectors with their original labels
#     # p.add_items(deleted_vectors, ids=np.array(delete_indices), replace_deleted=True)
#     p.add_items(deleted_vectors, replace_deleted=True)
#
#
#     # Perform k-NN search and measure recall and query time
#     labels, query_time = query_hnsw(p, queries, k)
#     recall = recall_score(indices_ground_truth, labels)
#
#     print(f"Iteration {iteration + 1}:")
#     print(f"RECALL: {recall:.4f}")
#     print(f"Query Time: {query_time:.4f} seconds\n")
