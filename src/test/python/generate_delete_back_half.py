import numpy as np
import hnswlib
import os

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

def create_initial_index(fvecs_path, index_dir):
    # Read all data
    all_data = list(fvecs_read(fvecs_path))
    total_vectors = len(all_data)

    # Split data into two halves
    first_half = np.array(all_data[:total_vectors // 2])
    second_half = np.array(all_data[total_vectors // 2:])

    # Create initial index with the first half
    dim = first_half.shape[1]
    num_elements = first_half.shape[0]

    # Initialize and create index
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=total_vectors, ef_construction=200, M=16, allow_replace_deleted=True)
    p.add_items(first_half)
    p.save_index(os.path.join(index_dir, "initial_index"))
    print("初始")
    return p, second_half

def iterative_deletion_and_addition(p, second_half, index_dir):
    num_elements_to_process = second_half.shape[0] // 5  # 100,000 elements per iteration

    for i in range(5):
        # Mark for deletion the next 100,000 elements from the first half
        delete_indices = list(range(i * num_elements_to_process, (i + 1) * num_elements_to_process))
        for label in delete_indices:
            p.mark_deleted(label)

        # Add the next 100,000 elements from the second half
        new_data = second_half[i * num_elements_to_process : (i + 1) * num_elements_to_process]
        p.add_items(new_data, replace_deleted=True)
        print(str(i)+"结束了")
        # Save the index after each iteration
        p.save_index(os.path.join(index_dir, f"index_iteration_{i+1}"))

# Usage example
fvecs_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs"
index_dir = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half"

# Step 1: Create initial index
index, second_half = create_initial_index(fvecs_path, index_dir)

# Step 2: Iteratively delete and add data, saving after each iteration
iterative_deletion_and_addition(index, second_half, index_dir)
