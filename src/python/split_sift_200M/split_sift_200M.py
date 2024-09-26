import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def read_bvecs_chunk(filename, start_idx, num_vectors):
    with open(filename, 'rb') as f:
        # Read the first vector to get the dimension
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Move to the starting point of the chunk
        f.seek(start_idx * (dim + 4))

        # Read the chunk of vectors
        vectors = np.fromfile(f, dtype=np.uint8, count=num_vectors * (dim + 4)).reshape(-1, dim + 4)

    # Skip the first 4 bytes (dimension) and return the vectors
    return vectors[:, 4:], dim

def save_fvecs_chunk(filename, data, dim):
    num_vectors = data.shape[0]
    with open(filename, 'ab') as f:
        for vector in data:
            # Convert the vector to float32 and write in fvecs format
            np.array([dim], dtype=np.int32).tofile(f)
            vector.astype(np.float32).tofile(f)

def process_chunk(input_path, output_file, start_idx, chunk_size):
    data, dim = read_bvecs_chunk(input_path, start_idx, chunk_size)
    save_fvecs_chunk(output_file, data, dim)

input_path = '/root/WorkSpace/dataset/sift/bigann_base.bvecs'
output_path = '/root/WorkSpace/dataset/sift/sift200M/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file = os.path.join(output_path, 'sift200M.fvecs')

# Number of vectors to extract (200 million)
num_vectors = 200000000
chunk_size = 10000000  # Process 1,000,000 vectors at a time
num_chunks = (num_vectors + chunk_size - 1) // chunk_size  # Ceiling division

# Remove the output file if it already exists to avoid appending to an old file
if os.path.exists(output_file):
    os.remove(output_file)

# Process chunks in parallel
with ThreadPoolExecutor() as executor:
    futures = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        current_chunk_size = min(chunk_size, num_vectors - start_idx)
        futures.append(executor.submit(process_chunk, input_path, output_file, start_idx, current_chunk_size))

    # Wait for all futures to complete
    for future in futures:
        future.result()

print(f"Extracted {num_vectors} vectors and saved to {output_file}")
