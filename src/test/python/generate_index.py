import numpy as np
import hnswlib

def fvecs_read(filename):
    """Reads .fvecs format file."""
    with open(filename, "rb") as f:
        while True:
            raw_dim = f.read(4)
            if len(raw_dim) < 4:
                break
            dim = np.frombuffer(raw_dim, dtype=np.int32, count=1)[0]
            raw_vec = f.read(dim * 4)
            if len(raw_vec) < dim * 4:
                break
            vec = np.frombuffer(raw_vec, dtype=np.float32, count=dim)
            yield vec

def create_index_from_half_data(fvecs_path, index_path):
    # 首先读取数据，确定总数量
    all_data = list(fvecs_read(fvecs_path))
    total_vectors = len(all_data)
    half_index = total_vectors // 2

    # 只取后一半的数据
    data = np.array(all_data[half_index:])

    # 创建一个hnswlib索引
    dim = data.shape[1]  # Dimension of the vectors
    num_elements = data.shape[0]  # Number of vectors in the second half

    # Initialize index
    p = hnswlib.Index(space='l2', dim=dim)
    # p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    p.init_index(max_elements=total_vectors, ef_construction=200, M=16)

    # Add data to the index
    p.add_items(data)

    # Save index
    p.save_index(index_path)


# 使用函数示例
create_index_from_half_data("/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs",
                            "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base_back_half.bin")



