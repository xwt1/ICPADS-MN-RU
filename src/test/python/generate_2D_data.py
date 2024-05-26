import numpy as np
import struct

def generate_dataset(num_vectors, filename1, filename2):
    # 创建一个空的数组来存储数据
    data = np.zeros((num_vectors, 2))

    # 生成x轴递增的值，确保不会有太大的跳跃
    data[:, 0] = np.linspace(0, 2000000, num=num_vectors)

    # 为y轴生成随机值
    data[:, 1] = np.random.uniform(0, 2000000, size=num_vectors)

    # 分割数据为两个部分
    first_half = data[:num_vectors//2]
    second_half = data[num_vectors//2:]

    # 将第一部分数据保存为.fvecs格式
    with open(filename1, 'wb') as f:
        for vector in first_half:
            # 写入向量的维度（此处为2）
            f.write(struct.pack('i', 2))
            # 写入向量的数据
            f.write(struct.pack('f' * 2, *vector))

    # 将第二部分数据保存为.fvecs格式
    with open(filename2, 'wb') as f:
        for vector in second_half:
            # 写入向量的维度（此处为2）
            f.write(struct.pack('i', 2))
            # 写入向量的数据
            f.write(struct.pack('f' * 2, *vector))

# 调用函数，生成数据集
generate_dataset(2000000, '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_update_base.fvecs',
                 '/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_update_add.fvecs')
