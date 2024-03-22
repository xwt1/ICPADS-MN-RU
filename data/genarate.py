import numpy as np
import argparse

def generate_unique_random_points(dimension, num_points, max_attempts=1000000):
    """
    Generate unique random points within a given dimension and number.
    This function attempts to generate a set of unique points up to max_attempts times.
    """
    unique_points = set()
    attempts = 0
    while len(unique_points) < num_points and attempts < max_attempts:
        # Generate a point
        point = tuple((np.random.rand(dimension) * 100).astype(np.float32))
        unique_points.add(point)
        attempts += 1

    if len(unique_points) < num_points:
        raise ValueError("Unable to generate the required number of unique points within the maximum number of attempts.")

    # Convert the set to a numpy array
    return np.array(list(unique_points), dtype=np.float32)

def generate_random_points_and_save_binary(dimension, num_points, file_path):
    """
    Generate random points of given dimension and number, ensuring uniqueness,
    and save to a binary file.
    """
    points = generate_unique_random_points(dimension, num_points)
    print(points[:min(5, len(points))])  # 打印前5个点来检查
    points.flatten().tofile(file_path)  # 以二进制格式保存

if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="Generate unique random points and save them in binary format.")
    parser.add_argument("dimension", type=int, help="Dimension of the points.")
    parser.add_argument("num_points", type=int, help="Number of unique points to generate.")
    parser.add_argument("file_path", type=str, help="Path to save the binary file.")

    # 解析命令行参数
    args = parser.parse_args()

    # 生成随机点并保存
    generate_random_points_and_save_binary(args.dimension, args.num_points, args.file_path)
