import numpy as np
import struct
import threading

# Function to load fvecs file
def load_fvecs(file_path):
    with open(file_path, 'rb') as f:
        data = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = int.from_bytes(dim_bytes, byteorder='little')
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            data.append(vec)
    return np.array(data)

# Function to save ivecs file
def save_ivecs(file_path, data):
    with open(file_path, 'wb') as f:
        for vec in data:
            k = len(vec)
            f.write(struct.pack('i', k))
            for val in vec:
                f.write(struct.pack('i', val))

# Function to compute k-nearest neighbors
def knn(data, queries, k):
    num_data = data.shape[0]
    num_queries = queries.shape[0]
    indices = np.zeros((num_queries, k), dtype=int)

    def knn_thread(start_idx, end_idx):
        for i in range(start_idx, end_idx):
            distances = np.linalg.norm(data - queries[i], axis=1)
            knn_indices = np.argpartition(distances, k)[:k]
            sorted_indices = knn_indices[np.argsort(distances[knn_indices])]
            indices[i] = sorted_indices

    num_threads = min(8, num_queries)  # Limiting to a maximum of 8 threads
    chunk_size = (num_queries + num_threads - 1) // num_threads
    threads = []

    for t in range(num_threads):
        start_idx = t * chunk_size
        end_idx = min(start_idx + chunk_size, num_queries)
        if start_idx < end_idx:
            thread = threading.Thread(target=knn_thread, args=(start_idx, end_idx))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    return indices

# Paths
data_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/gist/gist_base.fvecs'
query_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/gist/gist_query.fvecs'
output_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/gist/ground_truth_python.ivecs'
k = 100

# Load data and queries
data = load_fvecs(data_path)
queries = load_fvecs(query_path)

# Compute k-nearest neighbors
ground_truth = knn(data, queries, k)

# Save results
save_ivecs(output_path, ground_truth)
