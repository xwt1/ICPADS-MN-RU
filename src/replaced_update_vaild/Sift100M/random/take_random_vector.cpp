#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include "hnswlib/hnswlib.h"
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <unordered_map>
#include <chrono>

// Function to load bvecs in batch based on provided indices
std::vector<std::vector<float>> load_bvecs_batch(const std::string& filename, const std::vector<size_t>& indices, int dim) {
    std::vector<std::vector<float>> data;
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    size_t vector_size_in_bytes = sizeof(int) + sizeof(uint8_t) * dim;

    for (size_t idx : indices) {
        // Calculate the starting position of the corresponding vector
        size_t start_pos = idx * vector_size_in_bytes;

        // Move the file pointer to the specified position
        input.seekg(start_pos, std::ios::beg);
        if (input.fail()) {
            std::cerr << "无法定位到文件中的位置: " << idx << std::endl;
            exit(1);
        }

        int cur_dim = 0;
        input.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (cur_dim != dim) {
            std::cerr << "维度不匹配，预期: " << dim << ", 实际: " << cur_dim << std::endl;
            exit(1);
        }

        // Read vector data
        std::vector<uint8_t> vec_uint8(dim);
        input.read(reinterpret_cast<char*>(vec_uint8.data()), sizeof(uint8_t) * dim);
        if (input.fail()) {
            std::cerr << "读取向量失败，位置: " << idx << std::endl;
            exit(1);
        }

        // Convert uint8_t data to float
        std::vector<float> vec_float(dim);
        for (int d = 0; d < dim; ++d) {
            vec_float[d] = static_cast<float>(vec_uint8[d]);
        }

        data.push_back(std::move(vec_float));
    }

    input.close();
    return data;
}

// Function to load ivecs
std::vector<size_t> load_ivecs(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    std::vector<size_t> data;
    while (!input.eof()) {
        int size;
        input.read(reinterpret_cast<char*>(&size), sizeof(int));
        if (input.eof()) break;

        std::vector<size_t> vec(size);
        input.read(reinterpret_cast<char*>(vec.data()), sizeof(size_t) * size);
        data.insert(data.end(), vec.begin(), vec.end());
    }

    input.close();
    return data;
}

// Function to save fvecs in batches
void save_to_fvecs(const std::string& filename, const std::vector<std::vector<float>>& data, bool append = false) {
    std::ofstream output;
    if (append) {
        output.open(filename, std::ios::binary | std::ios::app);
    } else {
        output.open(filename, std::ios::binary);
    }
    if (!output) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        exit(1);
    }

    for (const auto& vec : data) {
        int dim = vec.size();
        output.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        output.write(reinterpret_cast<const char*>(vec.data()), sizeof(float) * dim);
    }

    output.close();
}

int main(int argc, char* argv[]) {
    std::string output_random_indice = "/root/WorkSpace/dataset/sift/sift200M/sift100M_random_data.ivecs";
    std::string take_vector_path = "/root/WorkSpace/dataset/sift/sift200M/take_random_vector.fvecs";
    std::string data_path = "/root/WorkSpace/dataset/sift/bigann_base.bvecs";

    size_t dim = 128; // Assuming dimensionality of vectors is 128

    // Load random indices
    std::vector<size_t> random_indices = load_ivecs(output_random_indice);

    // Extract and save vectors in batches of 100000
    size_t batch_size = 100000;
    size_t batch_count = 0;

    for (size_t i = 0; i < random_indices.size(); i += batch_size) {
        // Extract a batch of indices
        std::vector<size_t> batch_indices(random_indices.begin() + i, random_indices.begin() + std::min(i + batch_size, random_indices.size()));

        // Load the vectors for the current batch
        std::vector<std::vector<float>> batch_vectors = load_bvecs_batch(data_path, batch_indices, dim);

        // Save the current batch to the file
        save_to_fvecs(take_vector_path, batch_vectors, batch_count > 0);
        batch_count++;
        std::cout << "批次 " << batch_count << " 已保存." << std::endl;
    }

    std::cout << "所有提取的向量已保存到 fvecs 格式文件中." << std::endl;

    return 0;
}