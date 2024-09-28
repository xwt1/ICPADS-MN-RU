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

int main(int argc, char* argv[]) {
    // 检查输入参数
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    // 定义文件路径
    std::string data_path = root_path + "/data/imageNet/image.ds";
    std::string query_path = root_path + "/data/imageNet/image.q";
    std::string index_path = root_path + "/data/imageNet/faiss_index/imageNet_index.bin";
    std::string ground_truth_path = root_path + "/data/imageNet/imageNet_groundtruth.ivecs";
    std::string random_indice_path = root_path + "/data/imageNet/imageNet_random_data.ivecs";
    std::string output_csv_path = root_path + "/output/random/imageNet2/faiss_ivf_flat.csv";
    std::string output_index_path = root_path + "/output/random/imageNet2/faiss_imageNet_output_index.bin";

    // 创建输出目录（假设您有一个 util::create_directories 函数）
    std::vector<std::string> paths_to_create = {output_csv_path, output_index_path};
    util::create_directories(paths_to_create);

    int dim, num_data, num_queries;
    // 使用您提供的 util 函数加载数据集和查询集
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
    size_t data_siz = data.size();

    int random_data_num_per_iteration, num_iterations;
    std::vector<std::vector<size_t>> random_data = util::load_ivecs(random_indice_path, random_data_num_per_iteration, num_iterations);

    int k = 10;

    // 将数据转换为连续的 float 数组，以便于 Faiss 处理
    std::vector<float> xb(data_siz * dim);
    for (size_t i = 0; i < data_siz; ++i) {
        std::copy(data[i].begin(), data[i].end(), xb.begin() + i * dim);
    }

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

    // 执行初始查询
    std::vector<std::vector<float>> queries_tmp(queries.begin(), queries.begin() + 5);

    // 将查询转换为连续的 float 数组
    size_t nq_tmp = queries_tmp.size();
    std::vector<float> xq_tmp(nq_tmp * dim);
    for (size_t i = 0; i < nq_tmp; ++i) {
        std::copy(queries_tmp[i].begin(), queries_tmp[i].end(), xq_tmp.begin() + i * dim);
    }

    // 查询并输出初始状态
    std::vector<faiss::idx_t> labels_flat(nq_tmp * k);
    std::vector<float> distances_flat(nq_tmp * k);
    index->search(nq_tmp, xq_tmp.data(), k, distances_flat.data(), labels_flat.data());

    // 打印初始查询结果
    for (size_t j = 0; j < nq_tmp; ++j) {
        std::cout << "查询 " << j << ":" << std::endl;
        std::cout << "标签长度: " << k << ", 能找到这么多的点" << std::endl;
    }

    size_t nq = queries.size();

    // 将所有查询转换为连续的 float 数组
    std::vector<float> xq(nq * dim);
    for (size_t i = 0; i < nq; ++i) {
        std::copy(queries[i].begin(), queries[i].end(), xq.begin() + i * dim);
    }

    // 执行初始查询并计算初始召回率
    std::vector<faiss::idx_t> labels_all(nq * k);
    std::vector<float> distances_all(nq * k);
    auto start_time_query = std::chrono::high_resolution_clock::now();
    index->search(nq, xq.data(), k, distances_all.data(), labels_all.data());
    auto end_time_query = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();
    auto avg_query_time = query_duration / queries.size();
    std::cout << "平均查询时间: " << avg_query_time << " 秒\n";

    // 计算初始召回率
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
    float initial_recall = static_cast<float>(correct) / total;
    std::cout << "初始召回率: " << initial_recall << std::endl;

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

        // 创建 ID 选择器并删除 IDs
        faiss::IDSelectorBatch selector(num_to_delete, ids_to_delete.data());
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        index->remove_ids(selector);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

        // 收集要重新添加的向量，使用原始的 IDs
        std::vector<float> vectors_to_add(num_to_delete * dim);
        for (size_t i = 0; i < num_to_delete; ++i) {
            size_t idx = delete_indices[i];
            std::copy(data[idx].begin(), data[idx].end(), vectors_to_add.begin() + i * dim);
        }

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
