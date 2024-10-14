//
// Created by root on 5/30/24.
//
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
#include "util.h"

// 添加 get_bvecs_file_info 函数
void get_bvecs_file_info(const std::string& filename, int& dim, size_t& num_vectors) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    // 读取第一个向量的维度
    int temp_dim;
    input.read(reinterpret_cast<char*>(&temp_dim), sizeof(int));
    if (input.fail()) {
        std::cerr << "读取维度失败。" << std::endl;
        exit(1);
    }
    dim = temp_dim;

    // 获取文件大小
    input.seekg(0, std::ios::end);
    std::streampos file_size = input.tellg();

    // 计算向量数量
    size_t vector_size_in_bytes = sizeof(int) + sizeof(uint8_t) * dim;
    num_vectors = file_size / vector_size_in_bytes;

    input.close();
}

// 添加 load_bvecs_range 函数
std::vector<std::vector<float>> load_bvecs_range(const std::string& filename, size_t start_idx, size_t count, int dim) {
    std::vector<std::vector<float>> data;
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    size_t vector_size_in_bytes = sizeof(int) + sizeof(uint8_t) * dim;
    size_t start_pos = start_idx * vector_size_in_bytes;

    // 移动文件指针到起始位置
    input.seekg(start_pos, std::ios::beg);
    if (input.fail()) {
        std::cerr << "无法定位到文件中的起始位置。" << std::endl;
        exit(1);
    }

    for (size_t i = 0; i < count; ++i) {
        int cur_dim = 0;
        input.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (cur_dim != dim) {
            std::cerr << "维度不匹配，预期: " << dim << ", 实际: " << cur_dim << std::endl;
            exit(1);
        }
        std::vector<uint8_t> vec_uint8(dim);
        input.read(reinterpret_cast<char*>(vec_uint8.data()), sizeof(uint8_t) * dim);
        if (input.fail()) {
            std::cerr << "读取向量失败。" << std::endl;
            exit(1);
        }
        // 将 uint8_t 数据转换为 float
        std::vector<float> vec_float(dim);
        for (int d = 0; d < dim; ++d) {
            vec_float[d] = static_cast<float>(vec_uint8[d]);
        }
        data.push_back(std::move(vec_float));
    }

    input.close();
    return data;
}

// 添加 load_bvecs_batch 函数
std::vector<std::vector<float>> load_bvecs_batch(const std::string& filename, const std::vector<size_t>& indices, int dim) {
    std::vector<std::vector<float>> data;
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    size_t vector_size_in_bytes = sizeof(int) + sizeof(uint8_t) * dim;

    for (size_t idx : indices) {
        // 计算对应向量的起始位置
        size_t start_pos = idx * vector_size_in_bytes;

        // 移动文件指针到指定位置
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

        // 读取向量数据
        std::vector<uint8_t> vec_uint8(dim);
        input.read(reinterpret_cast<char*>(vec_uint8.data()), sizeof(uint8_t) * dim);
        if (input.fail()) {
            std::cerr << "读取向量失败，位置: " << idx << std::endl;
            exit(1);
        }

        // 将 uint8_t 数据转换为 float
        std::vector<float> vec_float(dim);
        for (int d = 0; d < dim; ++d) {
            vec_float[d] = static_cast<float>(vec_uint8[d]);
        }

        data.push_back(std::move(vec_float));
    }

    input.close();
    return data;
}

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];


    std::string data_path = root_path + "/sift/bigann_base.bvecs";
    std::string query_path = root_path + "/sift/sift200M/bigann_query.bvecs";
    std::string index_path = root_path + "/sift/sift200M/index/sift_100M_index.bin";
    std::string ground_truth_path = root_path + "/sift/sift200M/gnd/idx_100M.ivecs";
    std::string output_csv_path = root_path + "/output/random/sift200M/edge_connected_replaced_update8.csv";
    std::string output_index_path = root_path + "/output/random/sift200M/edge_connected_replaced_update8_sift100M_random_index.bin";

    std::string random_indice_path = "/root/WorkSpace/dataset/sift/sift200M/sift100M_random_data.ivecs";

    std::vector<std::string> paths_to_create ={output_csv_path,output_index_path};
    util::create_directories(paths_to_create);




//    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    int dim = 128, num_queries = 10000;
    std::vector<std::vector<float>> queries = load_bvecs_range(query_path, 0, num_queries, dim);


    // 获取数据集信息而不加载到内存中
    size_t num_data = 0;
    get_bvecs_file_info(data_path, dim, num_data);

    num_data = 100000000;
    size_t data_siz = 100000000;

    int random_data_num_per_iteration, num_iterations;
    std::vector<std::vector<size_t>> random_data = util::load_ivecs(random_indice_path,random_data_num_per_iteration,num_iterations);


    int k = 1000;

    // Initialize the HNSW index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    std::cout << "索引加载完毕 " << std::endl;
    // 设置查询参数`ef`
    int ef = 500;
    index.setEf(ef);

    int num_threads = 40;
    // Number of iterations for delete and re-add process
//    int num_iterations = 200;
//    std::random_device rd;
//    std::mt19937 gen(rd());

    // Perform initial brute-force k-NN search to get ground truth
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);


    // generate CSV file
    std::vector<std::vector <std::string>> header = {{"iteration_number" , "unreachable_points_number","recall","avg_delete_time",
                                                      "avg_add_time","avg_sum_delete_add_time","avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);


    size_t last_idx = 0;
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        int num_to_delete = random_data[iteration].size();
        std::vector<size_t> delete_indices(random_data[iteration].begin(),random_data[iteration].end());

        // Save the vectors and their labels to be deleted before deleting them
        std::vector<std::vector<float>> deleted_vectors = load_bvecs_batch(data_path, delete_indices, dim);
//        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));

//        for (size_t i = 0; i < delete_indices.size(); ++i) {
//            size_t idx = delete_indices[i];
//            deleted_vectors[i] = data[idx];
//        }

        auto start_time_delete = std::chrono::high_resolution_clock::now();
        util::markDeleteMultiThread(index, delete_indices, index_map, num_threads);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();


        // Re-add the deleted vectors with their original labels
        std::vector<size_t> new_indices(delete_indices.size());
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = index_map[delete_indices[i]];
            size_t new_idx = (idx < num_data) ? idx + num_data : idx - num_data;
            new_indices[i] = new_idx;
            index_map[delete_indices[i]] = new_idx;
        }

        auto start_time_add = std::chrono::high_resolution_clock::now();
        util::addPointsMultiThread(index, deleted_vectors, new_indices, num_threads);
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        // Perform k-NN search and measure recall and query time
        std::vector<std::vector<size_t>> labels;

        auto start_time_query = std::chrono::high_resolution_clock::now();
        util::query_hnsw(index, queries, k, num_threads, labels);
        auto end_time_query = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

        float recall = util::recall_score(ground_truth, labels, index_map, data_siz);

        auto avg_delete_time = delete_duration / num_to_delete;
        auto avg_add_time = add_duration / num_to_delete;
        auto avg_query_time = query_duration / queries.size();

        auto avg_sum_delete_add_time = avg_delete_time + avg_add_time;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Iteration " << iteration + 1 << ":\n";
        std::cout<<"删除了"<<num_to_delete<<"个点"<<std::endl;
        std::cout << "RECALL: " << recall << "\n";
        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Avg SUM Delete Add Time: " << avg_sum_delete_add_time << " seconds\n";

        std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+1);
        auto results = util::query_index(&index, queries_tmp, data_siz);
        std::unordered_map <size_t,bool> excluded_global_labels_all;
        for (size_t j = 0; j < queries_tmp.size(); ++j) {
            std::cout << "Query " << j << ":" << std::endl;
            std::cout << "Labels length: " << results[j].first.size() << ",只能找到这么多的点" << std::endl;
        }
        std::cout << "------------------------------------------------------------------" << std::endl;

        std::string iteration_string = std::to_string(iteration + 1);
        std::string unreachable_points_string = std::to_string(data_siz - results.front().first.size());
        std::string recall_string = std::to_string(recall);
        std::string avg_delete_time_string = std::to_string(avg_delete_time);
        std::string avg_add_time_string = std::to_string(avg_add_time);
        std::string avg_sum_delete_add_time_string = std::to_string(avg_sum_delete_add_time);
        std::string avg_query_time_string = std::to_string(avg_query_time);

        std::vector<std::vector <std::string>> result_data = {{iteration_string, unreachable_points_string,recall_string,
                                                               avg_delete_time_string,avg_add_time_string,avg_sum_delete_add_time_string,avg_query_time_string}};

        util::writeCSVApp(output_csv_path, result_data);
    }
    index.saveIndex(output_index_path);
    return  0;
}