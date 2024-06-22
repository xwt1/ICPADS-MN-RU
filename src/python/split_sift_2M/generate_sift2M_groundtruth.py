import numpy as np
import os
import argparse
import threading

def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    vectors = []
    i = 0
    while i < len(data):
        size = np.frombuffer(data[i:i+4], dtype=np.int32)[0]
        i += 4
        vector = np.frombuffer(data[i:i+size*4], dtype=np.float32)
        vectors.append(vector)
        i += size * 4
    return np.vstack(vectors)

def write_fvecs(file_path, vectors):
    with open(file_path, 'wb') as f:
        for vector in vectors:
            f.write(np.array([vector.size], dtype=np.int32).tobytes())
            f.write(vector.tobytes())

def knn_worker(start_idx, end_idx, queries, database, top_k, results):
    for i in range(start_idx, end_idx):
        distances = np.linalg.norm(database - queries[i], axis=1)
        knn_indices = np.argpartition(distances, top_k)[:top_k]
        sorted_indices = knn_indices[np.argsort(distances[knn_indices])]
        results[i] = sorted_indices

def brute_force_search(database, queries, top_k=100, num_threads=None):
    if num_threads is None:
        num_threads = os.cpu_count()  # 检测机器的CPU核心数
    num_queries = queries.shape[0]
    top_k_indices = np.zeros((num_queries, top_k), dtype=int)

    chunk_size = (num_queries + num_threads - 1) // num_threads
    threads = []

    for t in range(num_threads):
        start_idx = t * chunk_size
        end_idx = min(start_idx + chunk_size, num_queries)
        if start_idx < end_idx:
            thread = threading.Thread(target=knn_worker, args=(start_idx, end_idx, queries, database, top_k, top_k_indices))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    return top_k_indices

def main(root_path):
    first_half_file = os.path.join(root_path, 'data/sift_2M/sift_2M_first_half.fvecs')
    second_half_file = os.path.join(root_path, 'data/sift_2M/sift_2M_back_half.fvecs')
    query_file = os.path.join(root_path, 'data/sift_2M/bigann_query.fvecs')
    output_dir = os.path.join(root_path, 'data/sift_2M/groundTruth')

    # 读取向量和查询
    first_half_vectors = read_fvecs(first_half_file)
    second_half_vectors = read_fvecs(second_half_file)
    queries = read_fvecs(query_file)  # 读取fvecs格式的查询向量

    data_size = first_half_vectors.shape[0] * 2  # 因为合并后的总大小为2百万

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 计算groundTruth
    num_iterations = 10
    step_size = 100000  # 每轮前移10万

    for i in range(num_iterations):
        start_idx = i * step_size
        combined_vectors = np.vstack((first_half_vectors[start_idx:], second_half_vectors[:step_size]))

        ground_truth = brute_force_search(combined_vectors, queries, top_k=100)

        # groundTruth直接记录组合后的向量空间的索引
        output_file = os.path.join(output_dir, f'sift_2M_{i+1}.fvecs')
        write_fvecs(output_file, ground_truth)
        print(f"Iteration {i+1}: Saved ground truth to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ground truth using brute force search.')
    parser.add_argument('root_path', type=str, help='Root path for the data files')
    args = parser.parse_args()
    main(args.root_path)
