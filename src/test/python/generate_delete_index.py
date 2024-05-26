import numpy as np
import hnswlib

def fvecs_read(filename):
    """Reads .fvecs format file, handling cases where the buffer is smaller than expected."""
    with open(filename, "rb") as f:
        while True:
            raw_dim = f.read(4)  # Read the dimension size (4 bytes)
            if len(raw_dim) < 4:
                break  # If we can't read 4 bytes, we've hit EOF
            dim = np.frombuffer(raw_dim, dtype=np.int32, count=1)[0]
            raw_vec = f.read(dim * 4)
            if len(raw_vec) < dim * 4:
                break  # If there's not enough data left for a full vector, end reading
            vec = np.frombuffer(raw_vec, dtype=np.float32, count=dim)
            yield vec

def build_and_save_index(input_path, output_path):
    # 读取数据
    data = np.array(list(fvecs_read(input_path)))

    # 创建一个hnswlib索引
    dim = data.shape[1]  # 向量的维度
    num_elements = data.shape[0]  # 向量的数量

    # 初始化索引
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # 添加数据到索引
    p.add_items(data)

    # 标记删除数据集的前一半
    num_to_delete = num_elements // 2
    for i in range(num_to_delete):
        p.mark_deleted(i)

    # 保存修改后的索引
    p.save_index(output_path)

# 调用函数
build_and_save_index("/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs",
                     "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base_deleted_half.bin")
