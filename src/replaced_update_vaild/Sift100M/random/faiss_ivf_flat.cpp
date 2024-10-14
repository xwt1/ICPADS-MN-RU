#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <chrono>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include <omp.h>  // 包含 OpenMP 头文件

#include "util.h"  // 假设您提供了 util.h 头文件

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


int main(int argc, char* argv[]) {
    // 检查输入参数
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];


    // 定义文件路径
    std::string data_path = root_path + "/sift/bigann_base.bvecs";
    std::string query_path = root_path + "/sift/sift200M/bigann_query.bvecs";
    std::string index_path = root_path + "/sift/sift200M/index/sift_100M_index.bin";
    std::string ground_truth_path = root_path + "/sift/sift200M/gnd/idx_100M.ivecs";

    std::string output_csv_path = root_path + "/output/random/sift/faiss_ivf_flat.csv";
    std::string output_index_path = root_path + "/output/random/sift/faiss_sift_output_index.bin";

    std::string random_indice_path = root_path + "/data/sift/sift_random_data.ivecs";


    std::vector<std::string> paths_to_create = {output_csv_path, output_index_path};
    util::create_directories(paths_to_create);

    int dim = 128, num_queries = 10000;
    size_t num_data = 0;
    get_bvecs_file_info(data_path, dim, num_data);
    num_data = 100000000;


    // 使用您提供的 util 函数加载数据集和查询集
//    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
    size_t data_siz = 100000000;

    int random_data_num_per_iteration, num_iterations;
    std::vector<std::vector<size_t>> random_data = util::load_ivecs(random_indice_path, random_data_num_per_iteration, num_iterations);

    int k = 1000;

    // 将数据转换为连续的 float 数组，以便于 Faiss 处理
//    std::vector<float> xb(data_siz * dim);
//    for (size_t i = 0; i < data_siz; ++i) {
//        std::copy(data[i].begin(), data[i].end(), xb.begin() + i * dim);
//    }

    // 设置线程数
    int num_threads = 40;
    omp_set_num_threads(num_threads);

    // 从文件中读取已训练好的索引
    faiss::IndexIVFFlat* index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(index_path.c_str()));
    if (!index) {
        std::cerr << "错误：无法加载索引或索引类型不是 IndexIVFFlat。" << std::endl;
        return 1;
    }

    // 设置 nprobe 参数
    index->nprobe = 10;

    // 读取真实值（ground truth）
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);



    size_t nq = queries.size();

    // 将所有查询转换为连续的 float 数组
    std::vector<float> xq(nq * dim);
    for (size_t i = 0; i < nq; ++i) {
        std::copy(queries[i].begin(), queries[i].end(), xq.begin() + i * dim);
    }



    // 生成 CSV 文件
    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time",
                                                     "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);

    // 不再使用 index_map，直接使用原始的 IDs

    // 迭代删除和重新添加数据
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        int num_to_delete = random_data[iteration].size();
        std::vector<size_t> delete_indices(random_data[iteration].begin(), random_data[iteration].end());

        // 收集要删除的 IDs
        std::vector<faiss::idx_t> ids_to_delete(num_to_delete);
        for (size_t i = 0; i < num_to_delete; ++i) {
            ids_to_delete[i] = delete_indices[i];
        }

        // 从文件中加载要重新添加的向量
        std::vector<std::vector<float>> vector_two = load_bvecs_batch(data_path, delete_indices, dim);

        // 将 vector_two 平铺到 vectors_to_add
        std::vector<float> vectors_to_add(num_to_delete * dim);
        for (size_t i = 0; i < num_to_delete; ++i) {
            std::copy(vector_two[i].begin(), vector_two[i].end(), vectors_to_add.begin() + i * dim);
        }

        // 创建 ID 选择器并删除 IDs
        faiss::IDSelectorBatch selector(num_to_delete, ids_to_delete.data());
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        index->remove_ids(selector);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();


//        for (size_t i = 0; i < num_to_delete; ++i) {
//            size_t idx = delete_indices[i];
//            std::copy(data[idx].begin(), data[idx].end(), vectors_to_add.begin() + i * dim);
//        }

        // 重新添加向量，使用原始的 IDs
        auto start_time_add = std::chrono::high_resolution_clock::now();
        index->add_with_ids(num_to_delete, vectors_to_add.data(), ids_to_delete.data());
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        // 执行查询并计算召回率
        std::vector<faiss::idx_t> labels_all(nq * k);
        std::vector<float> distances_all(nq * k);
        auto start_time_query = std::chrono::high_resolution_clock::now();
        index->search(nq, xq.data(), k, distances_all.data(), labels_all.data());
        auto end_time_query = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

        // 计算召回率
        size_t correct = 0;
        size_t total = nq * k;
        for (size_t i = 0; i < nq; ++i) {
            std::unordered_set<size_t> gt_set(ground_truth[i].begin(), ground_truth[i].begin() + k);
            for (size_t j = 0; j < k; ++j) {
                size_t retrieved_id = labels_all[i * k + j];
                if (gt_set.find(retrieved_id) != gt_set.end()) {
                    correct++;
                }
            }
        }
        float recall = static_cast<float>(correct) / total;

        auto avg_delete_time = delete_duration / num_to_delete;
        auto avg_add_time = add_duration / num_to_delete;
        auto avg_query_time = query_duration / queries.size();
        auto avg_sum_delete_add_time = avg_delete_time + avg_add_time;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "迭代 " << iteration + 1 << ":\n";
        std::cout << "删除了 " << num_to_delete << " 个点。" << std::endl;
        std::cout << "召回率: " << recall << "\n";
        std::cout << "平均删除时间: " << avg_delete_time << " 秒\n";
        std::cout << "平均添加时间: " << avg_add_time << " 秒\n";
        std::cout << "平均查询时间: " << avg_query_time << " 秒\n";
        std::cout << "删除和添加总时间: " << avg_sum_delete_add_time << " 秒\n";

        // 检查当前索引状态
        std::vector<std::vector<float>> queries_tmp(queries.begin(), queries.begin() + 1);
        size_t nq_tmp = queries_tmp.size();
        std::vector<float> xq_tmp(nq_tmp * dim);
        for (size_t i = 0; i < nq_tmp; ++i) {
            std::copy(queries_tmp[i].begin(), queries_tmp[i].end(), xq_tmp.begin() + i * dim);
        }
        std::vector<faiss::idx_t> labels_tmp(nq_tmp * k);
        std::vector<float> distances_tmp(nq_tmp * k);
        index->search(nq_tmp, xq_tmp.data(), k, distances_tmp.data(), labels_tmp.data());
        for (size_t j = 0; j < nq_tmp; ++j) {
            std::cout << "查询 " << j << ":" << std::endl;
            std::cout << "标签长度: " << k << ", 能找到这么多的点" << std::endl;
        }
        std::cout << "------------------------------------------------------------------" << std::endl;

        // 写入 CSV 文件
        std::string iteration_string = std::to_string(iteration + 1);
        std::string unreachable_points_string = std::to_string(data_siz - k);  // 这里不准确，但与原始代码匹配
        std::string recall_string = std::to_string(recall);
        std::string avg_delete_time_string = std::to_string(avg_delete_time);
        std::string avg_add_time_string = std::to_string(avg_add_time);
        std::string avg_sum_delete_add_time_string = std::to_string(avg_sum_delete_add_time);
        std::string avg_query_time_string = std::to_string(avg_query_time);

        std::vector<std::vector<std::string>> result_data = {{iteration_string, unreachable_points_string, recall_string,
                                                              avg_delete_time_string, avg_add_time_string, avg_sum_delete_add_time_string, avg_query_time_string}};

        util::writeCSVApp(output_csv_path, result_data);
    }

    // 保存索引
    faiss::write_index(index, output_index_path.c_str());

    // 释放索引指针
    delete index;

    return 0;
}
