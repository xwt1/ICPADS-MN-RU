import numpy as np
import os
import time
import multiprocessing

def read_bvecs_batch(file_path, start_idx, end_idx, batch_size):
    """分批读取bvecs文件的指定范围"""
    with open(file_path, 'rb') as f:
        dimension = int(np.fromfile(f, dtype='int32', count=1)[0])
        vector_size = 4 + dimension  # 每个向量的字节数
        f.seek(0)  # 重置文件指针

        # 计算起始偏移量
        offset_bytes = start_idx * vector_size
        f.seek(offset_bytes, 1)

        total_vectors = end_idx - start_idx
        for offset in range(0, total_vectors, batch_size):
            vectors_to_read = min(batch_size, total_vectors - offset)
            data = np.fromfile(f, dtype='uint8', count=vector_size * vectors_to_read)
            if data.size != vector_size * vectors_to_read:
                break  # 文件读取完成
            data = data.reshape(-1, vector_size)
            vectors = data[:, 4:].astype('float32')
            yield vectors

def read_fvecs(file_path):
    """读取fvecs文件"""
    with open(file_path, 'rb') as f:
        data = []
        while True:
            header = f.read(4)
            if not header:
                break
            dim = np.frombuffer(header, dtype='int32')[0]
            vector = np.frombuffer(f.read(dim * 4), dtype='float32')
            data.append(vector)
        return np.array(data)

def save_ivecs(file_path, data):
    """保存ivecs文件"""
    with open(file_path, 'wb') as f:
        for vec in data:
            dim = np.array([len(vec)], dtype='int32')
            dim.tofile(f)
            np.array(vec, dtype='int32').tofile(f)

def compute_ground_truth_worker(args):
    """计算一部分查询向量的K近邻"""
    query_indices, query_vectors, base_path, total_base_vectors, K = args
    batch_size = 100_000  # 基向量的子批处理大小，可根据内存调整

    num_queries = query_vectors.shape[0]

    # 初始化每个查询的距离和索引
    topk_distances = [np.full(K, np.inf, dtype='float32') for _ in range(num_queries)]
    topk_indices = [np.full(K, -1, dtype='int32') for _ in range(num_queries)]

    base_idx = 0  # 基向量全局索引

    # 分批读取基向量
    for base_batch in read_bvecs_batch(base_path, 0, total_base_vectors, batch_size):
        num_base = base_batch.shape[0]

        # 对于每个查询向量，单独计算与当前基向量批次的距离
        for i in range(num_queries):
            query = query_vectors[i]

            # 计算距离
            dists = np.linalg.norm(base_batch - query, axis=1)

            # 合并当前的距离和索引
            combined_dists = np.concatenate((topk_distances[i], dists))
            combined_indices = np.concatenate((topk_indices[i], np.arange(base_idx, base_idx + num_base)))

            # 获取最小的K个距离及其索引
            idx = np.argpartition(combined_dists, K - 1)[:K]
            topk_distances[i] = combined_dists[idx]
            topk_indices[i] = combined_indices[idx]

        base_idx += num_base  # 更新基向量全局索引

    # 对每个查询的结果进行排序
    for i in range(num_queries):
        idx = np.argsort(topk_distances[i])
        topk_distances[i] = topk_distances[i][idx]
        topk_indices[i] = topk_indices[i][idx]

    return (query_indices, topk_indices)

def compute_ground_truth(query_vectors, base_path, total_base_vectors, K, output_path, num_processes):
    """计算精确的K近邻（多进程版本）"""
    num_queries = query_vectors.shape[0]

    print("开始计算...")
    start_time = time.time()

    # 将查询向量分成多个子集，每个进程处理一个子集
    query_indices_list = np.array_split(np.arange(num_queries), num_processes)
    query_vectors_list = np.array_split(query_vectors, num_processes)

    # 准备参数列表
    args_list = [(query_indices_list[i], query_vectors_list[i], base_path, total_base_vectors, K) for i in range(num_processes)]

    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)

    # 计算
    results = pool.map(compute_ground_truth_worker, args_list)

    pool.close()
    pool.join()

    # 合并结果
    topk_indices_all = [None] * num_queries
    for query_indices, topk_indices in results:
        for idx, indices in zip(query_indices, topk_indices):
            topk_indices_all[idx] = indices

    total_time = time.time() - start_time
    print(f"计算完成，耗时 {total_time // 3600:.0f}小时 {(total_time % 3600) // 60:.0f}分钟 {total_time % 60:.2f}秒")

    # 保存结果
    save_ivecs(output_path, topk_indices_all)
    print(f"已将Ground Truth保存到 {output_path}")

# 主程序
if __name__ == '__main__':
    # 参数设置
    base_path = '/root/WorkSpace/dataset/sift/bigann_base.bvecs'
    total_base_vectors = 100_000_000  # 前1亿个基向量
    query_path = '/root/WorkSpace/dataset/sift/sift200M/sift_query.fvecs'
    output_path = '/root/WorkSpace/dataset/sift/sift200M/sift_ground_truth.ivecs'
    K = 10
    num_processes = 48  # 设置进程数，可根据CPU核心数调整

    # 加载查询向量
    print("正在加载查询向量...")
    query_vectors = read_fvecs(query_path)
    num_queries = query_vectors.shape[0]
    print(f"加载查询向量完成，共 {num_queries} 个向量")

    # 计算Ground Truth
    compute_ground_truth(query_vectors, base_path, total_base_vectors, K, output_path, num_processes)
