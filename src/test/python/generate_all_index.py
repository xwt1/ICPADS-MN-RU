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

def create_index_with_all_data(fvecs_path, index_path):
    # 读取全部数据
    all_data = list(fvecs_read(fvecs_path))
    data = np.array(all_data)

    # 创建一个hnswlib索引
    dim = data.shape[1]  # Dimension of the vectors
    num_elements = data.shape[0]  # Total number of vectors

    # Initialize index
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # p.set_num_threads(48)

    # Add all data to the index
    p.add_items(data)

    # Save index
    p.save_index(index_path)

# 使用函数示例
# create_index_with_all_data("/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs",
#                            "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base_all.bin")

create_index_with_all_data("/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_update_base.fvecs",
                           "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_index/delete_update.bin")