//
// Created by root on 5/29/24.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

// Load fvecs file format
std::vector<float> load_fvecs(const std::string& filename, int& dim, int& num) {
    std::ifstream input(filename, std::ios::binary);
    input.read(reinterpret_cast<char*>(&dim), sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num = file_size / ((dim + 1) * sizeof(float));
    input.seekg(0, std::ios::beg);

    std::vector<float> data(num * dim);
    for (int i = 0; i < num; ++i) {
        input.ignore(sizeof(int)); // ignore dimension
        input.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
    }

    return data;
}


// 函数：计算两个向量之间的欧氏距离的平方
float euclidean_distance_squared(const float* a, const float* b, int dim) {
    float dist = 0;
    for (int d = 0; d < dim; ++d) {
        float diff = a[d] - b[d];
        dist += diff * diff;
    }
    return dist;
}

// 函数：使用暴力搜索算法找到每个查询点的k个最近邻
std::vector<std::vector<size_t>> brute_force_knn(const std::vector<float>& data, const std::vector<float>& queries, int dim, int k) {
    size_t num_data = data.size() / dim;
    size_t num_queries = queries.size() / dim;
    std::vector<std::vector<size_t>> indices(num_queries, std::vector<size_t>(k));

    for (size_t i = 0; i < num_queries; ++i) {
        std::vector<std::pair<float, size_t>> distances(num_data);
        for (size_t j = 0; j < num_data; ++j) {
            float dist = euclidean_distance_squared(&data[j * dim], &queries[i * dim], dim);
            distances[j] = { dist, j };
        }
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (int n = 0; n < k; ++n) {
            indices[i][n] = distances[n].second;
        }
    }

    return indices;
}

// 函数：将最近邻索引保存到ivecs格式文件中
void save_ground_truth_ivecs(const std::vector<std::vector<size_t>>& indices, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (const auto& vec : indices) {
        int dim = vec.size();
        outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        std::vector<int> int_vec(vec.begin(), vec.end());
        outfile.write(reinterpret_cast<const char*>(int_vec.data()), dim * sizeof(int));
    }

    outfile.close();
}

std::vector<std::vector<size_t>> load_ivecs_indices(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<size_t>> indices;
    while (input.peek() != EOF) {
        int dim;
        input.read(reinterpret_cast<char*>(&dim), sizeof(int));

        std::vector<int> vec(dim);
        input.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        indices.emplace_back(vec.begin(), vec.end());
    }

    input.close();
    return indices;
}

int main() {
    std::string data_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
    std::string query_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_query.fvecs";

    int dim, num_data, num_queries;

    int k = 5;     // 最近邻个数

    std::vector<float> data = load_fvecs(data_path, dim, num_data);
    std::vector<float> queries = load_fvecs(query_path, dim, num_queries);

    size_t data_siz = data.size() / dim;

//    int max_queries = 10;
//    num_queries = std::min(num_queries, max_queries);
//    queries.resize(num_queries * dim);

    // 计算最近邻
    auto indices = brute_force_knn(data, queries, dim, k);

    // 保存到ground_truth文件
    save_ground_truth_ivecs(indices, "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/freshdiskann_prove/sift1M_top5_ground_truth/ground_truth.ivecs");

    std::cout<<"生成结束"<<std::endl;
    return 0;
}


//int main() {
//    // 示例ground_truth文件路径
//    std::string ground_truth_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/freshdiskann_prove/sift1M_top5_ground_truth/ground_truth.ivecs";
//
////    std::string ground_truth_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_groundtruth.ivecs";
//    // 加载ground_truth文件中的索引
//    std::vector<std::vector<size_t>> indices = load_ivecs_indices(ground_truth_path);
//
//    int max_queries = 2;
//
//    indices.resize(max_queries * 128);
//
//    // 输出加载的索引信息
//    for (const auto& vec : indices) {
//        for (size_t idx : vec) {
//            std::cout << idx << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    return 0;
//}


