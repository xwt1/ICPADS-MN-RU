import numpy as np

def read_bvecs(file_path):
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            header = f.read(4)
            if not header:
                break
            dim = np.frombuffer(header, dtype=np.int32)[0]
            vector_bytes = f.read(dim)
            if len(vector_bytes) != dim:
                break
            vector = np.frombuffer(vector_bytes, dtype=np.uint8).astype(np.float32)
            vectors.append(vector)
    return np.vstack(vectors)

def write_fvecs(file_path, vectors):
    with open(file_path, 'wb') as f:
        for vector in vectors:
            f.write(np.array([vector.size], dtype=np.int32).tobytes())
            f.write(vector.tobytes())

# 文件路径
root_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement'

bvecs_file = root_path + '/data/sift_2M/bigann_query.bvecs'
fvecs_file = root_path + '/data/sift_2M/bigann_query.fvecs'

# 读取bvecs文件
vectors = read_bvecs(bvecs_file)

# 写入fvecs文件
write_fvecs(fvecs_file, vectors)

print(f"Converted {bvecs_file} to {fvecs_file}")
