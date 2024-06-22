import numpy as np

def read_bvecs(file_path, num_vectors, dim=128):
    """Read bvecs format file."""
    with open(file_path, 'rb') as f:
        vectors = np.fromfile(f, dtype=np.uint8, count=num_vectors * (dim + 1)).reshape(-1, dim + 1)[:, 1:].astype(np.float32)
    return vectors

def write_fvecs(file_path, vectors):
    """Write fvecs format file."""
    with open(file_path, 'wb') as f:
        for vector in vectors:
            f.write(np.array([vector.size], dtype=np.int32).tobytes())
            f.write(vector.tobytes())

# 文件路径

root_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement'

input_file = '/root/sift1B/bigann_base.bvecs'
first_half_output_file = root_path + '/data/sift_2M/sift_2M_first_half.fvecs'
second_half_output_file = root_path + '/data/sift_2M/sift_2M_back_half.fvecs'

# 提取向量
total_vectors_to_extract = 2_000_000
half_vectors = total_vectors_to_extract // 2

# 读取前200万个向量
vectors = read_bvecs(input_file, total_vectors_to_extract)

# 分别写入前一百万个和后一百万个向量
write_fvecs(first_half_output_file, vectors[:half_vectors])
write_fvecs(second_half_output_file, vectors[half_vectors:])
