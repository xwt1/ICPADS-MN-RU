import numpy as np
import diskannpy as dap
from pathlib import Path
from typing import Union

def load_fvecs(file_path):
    data = []
    with open(file_path, 'rb') as f:
        while True:
            dim = np.fromfile(f, dtype=np.int32, count=1)
            if not dim.size:
                break
            vec = np.fromfile(f, dtype=np.float32, count=dim[0])
            data.append(vec)
    return np.array(data, dtype=np.float32)

def build_diskann_index(
        data_path: str,
        index_directory: str,
        distance_metric: str = "l2",
        complexity: int = 100,
        graph_degree: int = 60,
        num_threads: int = 4
):
    # 加载数据
    data = load_fvecs(data_path)

    # 设置索引目录
    index_directory_path = Path(index_directory)
    index_directory_path.mkdir(parents=True, exist_ok=True)

    # 构建索引
    dap.build_memory_index(
        data=data,
        distance_metric=distance_metric,
        index_directory=str(index_directory_path),
        complexity=complexity,
        graph_degree=graph_degree,
        num_threads=num_threads
    )
    print("Index building completed.")

# 源数据路径
data_path = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_base.fvecs'
# 索引保存路径
index_directory = '/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/freshdiskann/diskannindex.bin'

# 构建索引
build_diskann_index(
    data_path=data_path,
    index_directory=index_directory,
    distance_metric="l2",  # 可以根据需要修改为"mips"或"cosine"
    complexity=100,
    graph_degree=60,
    num_threads=48
)
