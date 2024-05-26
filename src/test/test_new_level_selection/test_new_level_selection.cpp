#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "hnswlib/hnswlib.h"

// 读取fvecs文件
std::vector<float> read_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    input.read(reinterpret_cast<char*>(&dim), sizeof(int));

    std::vector<float> data;
    float value;
    size_t vector_size = dim * sizeof(float);
    while (input.read(reinterpret_cast<char*>(&value), sizeof(float))) {
        data.push_back(value);
        if (data.size() % dim == 0) {
            input.seekg(sizeof(int), std::ios::cur);
        }
    }
    num_vectors = data.size() / dim;
    return data;
}

int main() {
//    const std::string file_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
    const std::string file_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift_1w/siftsmall_base.fvecs";
    size_t num_vectors = 0;
    size_t dim = 0;

    // 读取数据
    std::vector<float> data = read_fvecs(file_path, num_vectors, dim);

    // 创建HNSW索引
    size_t max_elements = num_vectors;
    size_t M = 16;  // 内部参数
    size_t ef_construction = 200;  // 内部参数

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, max_elements, M, ef_construction);

    // 添加数据到索引并统计时间
    for (size_t i = 0; i < num_vectors; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        index.addPoint(data.data() + i * dim, i);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Index: " << i << ", Time: " << elapsed.count() << " seconds" << std::endl;
    }

    // 保存索引到文件
    index.saveIndex("/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/new_level_selection_index/sift_index_new_level_selection_2.bin");

    std::cout << "HNSW索引已成功构建并保存到'sift_index_new_level_selection.bin'。" << std::endl;

    return 0;
}
