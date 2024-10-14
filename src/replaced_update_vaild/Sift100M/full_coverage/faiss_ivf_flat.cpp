#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <omp.h>  // Include OpenMP header
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include "faiss/AutoTune.h"
#include "faiss/IndexIDMap.h"
#include "faiss/utils/random.h"
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/sift/bigann_base.bvecs";
    std::string query_path = root_path + "/sift/sift200M/bigann_query.bvecs";
    std::string index_path = root_path + "/sift/sift200M/index/faiss_ivf_index.bin";
    std::string ground_truth_path = root_path + "/sift/sift200M/gnd/idx_100M.ivecs";
    std::string output_csv_path = root_path + "/output/full_coverage/sift200M/faiss_ivf_flat.csv";
    std::string output_index_path = root_path + "/output/full_coverage/sift200M/faiss_sift100M_output_index.bin";

    std::vector<std::string> paths_to_create = {output_csv_path, output_index_path};
    util::create_directories(paths_to_create);

    // 获取数据集的维度和数量，而不加载全部数据
    int dim;
    size_t num_data;
    get_bvecs_file_info(data_path, dim, num_data);
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
    // 加载查询数据
    size_t num_queries = 10000;  // 假设查询集大小为 10,000
    std::vector<std::vector<float>> queries = load_bvecs_range(query_path, 0, num_queries, dim);

    // 将查询数据转换为 1D 数组
    std::vector<float> queries_1d(num_queries * dim);
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            queries_1d[i * dim + j] = queries[i][j];
        }
    }

    dim =128;
    int k = 1000;  // 最近邻数量
    num_data = 100000000;  // 如果您的数据集大小就是 1 亿
    int num_threads = 40;  // 设置线程数量
    omp_set_num_threads(num_threads);

    // 加载 Faiss IVF-Flat 索引
    faiss::IndexIVFFlat* index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(index_path.c_str()));
    if (!index) {
        std::cerr << "Failed to load Faiss index" << std::endl;
        return 1;
    }

    index->nprobe = 100;  // 设置搜索时的探测数

    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    std::cout << "Index loaded successfully." << std::endl;

    int num_iterations = 100;
    size_t last_idx = 0;
    size_t batch_size = 1000000;  // 每次加载 1 百万个向量

    // 生成 CSV 文件头
    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time", "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        std::unordered_set<size_t> delete_indices_set;

        // 计算删除数据的起始索引
        size_t start_idx = last_idx;
        int num_to_delete = num_data / num_iterations;
//        int num_to_delete = 10000;

        last_idx = start_idx + num_to_delete;

        // 加载要删除的向量
        std::vector<std::vector<float>> deleted_vectors = load_bvecs_range(data_path, start_idx, num_to_delete, dim);

        // 删除和重新添加操作
        std::vector<faiss::idx_t> delete_ids(num_to_delete);
        for (size_t i = 0; i < num_to_delete; ++i) {
            delete_ids[i] = static_cast<faiss::idx_t>(start_idx + i);
        }

        // 删除点
        faiss::IDSelectorArray selector(delete_ids.size(), delete_ids.data());
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        index->remove_ids(selector);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

        std::cout<<"delete end"<<std::endl;

        // 将删除的点重新添加
        std::vector<float> flat_deleted_vectors;
        for (const auto& vec : deleted_vectors) {
            flat_deleted_vectors.insert(flat_deleted_vectors.end(), vec.begin(), vec.end());
        }

        auto start_time_add = std::chrono::high_resolution_clock::now();
        index->add_with_ids(delete_ids.size(), flat_deleted_vectors.data(), delete_ids.data());
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        std::cout<<"add end"<<std::endl;

        // 执行查询并计算召回率
        std::vector<faiss::idx_t> I(num_queries * k);
        std::vector<float> D(num_queries * k);

        auto start_time_query = std::chrono::high_resolution_clock::now();
        index->search(num_queries, queries_1d.data(), k, D.data(), I.data());
        auto end_time_query = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

        // 计算召回率
        std::vector<std::vector<size_t>> predictions(num_queries, std::vector<size_t>(k));
        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < k; ++j) {
                predictions[i][j] = static_cast<size_t>(I[i * k + j]);
            }
        }

        // 按照手动计算召回率的方式
        size_t total_recall_hits = 0;
        size_t total_possible = num_queries * k;

        for (size_t i = 0; i < num_queries; ++i) {
            const auto& gt = ground_truth[i];
            const auto& pred = predictions[i];
            std::unordered_set<size_t> gt_set(gt.begin(), gt.end());

            for (size_t j = 0; j < k; ++j) {
                if (gt_set.find(pred[j]) != gt_set.end()) {
                    total_recall_hits++;
                }
            }
        }

        float recall = static_cast<float>(total_recall_hits) / total_possible;

        double avg_delete_time = delete_duration / num_to_delete;
        double avg_add_time = add_duration / num_to_delete;
        double avg_query_time = query_duration / num_queries;
        double avg_sum_delete_add_time = avg_delete_time + avg_add_time;

        std::cout << "Iteration " << iteration + 1 << ":\n";
        std::cout << "Recall: " << recall << "\n";
        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Avg SUM Delete Add Time: " << avg_sum_delete_add_time << " seconds\n";

        // 记录结果到 CSV 文件
        std::vector<std::vector<std::string>> result_data = {{std::to_string(iteration + 1), "0", std::to_string(recall),
                                                              std::to_string(avg_delete_time), std::to_string(avg_add_time),
                                                              std::to_string(avg_sum_delete_add_time), std::to_string(avg_query_time)}};
        util::writeCSVApp(output_csv_path, result_data);
    }

    // 保存更新后的索引
    faiss::write_index(index, output_index_path.c_str());

    return 0;
}



////
//// Created by root on 5/30/24.
////
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <random>
//#include <thread>
//#include <atomic>
//#include <mutex>
//#include <chrono>
//#include <omp.h>  // Include OpenMP header
//#include "faiss/IndexIVFFlat.h"
//#include "faiss/index_io.h"
//#include "faiss/AutoTune.h"
//#include "faiss/IndexIDMap.h"
//#include "faiss/utils/random.h"
//#include "util.h"
//
//int main(int argc, char* argv[]) {
//    if (argc < 2) {
//        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
//        return 1;
//    }
//    std::string root_path = argv[1];
//
//    std::string data_path = root_path + "/data/gist/gist_base.fvecs";
//    std::string query_path = root_path + "/data/gist/gist_query.fvecs";
//    std::string index_path = root_path + "/data/gist/faiss_index/gist_index.bin";
//    std::string ground_truth_path = root_path + "/data/gist/gist_groundtruth.ivecs";
//    std::string output_csv_path = root_path + "/output/full_coverage/gist/faiss_ivf_flat.csv";
//    std::string output_index_path = root_path + "/output/full_coverage/gist/faiss_gist_output_index.bin";
//
//    std::vector<std::string> paths_to_create = {output_csv_path, output_index_path};
//    util::create_directories(paths_to_create);
//
//    int dim, num_data, num_queries;
//    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
//    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
//
//    int k = 100;
//    int num_threads = 40;  // Set your desired number of threads
//
//    // Set the number of threads for Faiss operations
//    omp_set_num_threads(num_threads);
//
//    // Load Faiss IVF-Flat index
//    faiss::IndexIVFFlat* index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(index_path.c_str()));
//
//    if (!index) {
//        std::cerr << "Failed to load Faiss index" << std::endl;
//        return 1;
//    }
//
//    std::cout << index->ntotal << std::endl;
//
//    index->nprobe = 200; // Set number of probes
//
//    std::unordered_map<size_t, size_t> index_map;
//    for (size_t i = 0; i < num_data; ++i) {
//        index_map[i] = i;
//    }
//
//    std::cout << "Index loaded successfully." << std::endl;
//
//    // Load ground truth
//    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
//
//    // Perform an initial k-NN search using Faiss search to calculate initial recall
//    std::vector<faiss::idx_t> I(num_queries * k);
//    std::vector<float> D(num_queries * k);
//
//    auto start_time_query = std::chrono::high_resolution_clock::now();
//    index->search(num_queries, queries[0].data(), k, D.data(), I.data());
//    auto end_time_query = std::chrono::high_resolution_clock::now();
//    auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();
//
//    // Reshape Faiss search results for recall calculation
//    size_t total_recall_hits = 0;
//    for (size_t i = 0; i < num_queries; ++i) {
//        std::unordered_set<int> gt_set(ground_truth[i].begin(), ground_truth[i].begin() + k);
//        for (size_t j = 0; j < k; ++j) {
//            if (gt_set.find(I[i * k + j]) != gt_set.end()) {
//                total_recall_hits++;
//            }
//        }
//    }
//
//    float initial_recall = static_cast<float>(total_recall_hits) / (num_queries * k);
//    std::cout << "Initial Recall: " << initial_recall << std::endl;
//
//    // Generate CSV file header
//    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time", "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
//    util::writeCSVOut(output_csv_path, header);
//
//    size_t last_idx = 0;
//    int num_iterations = 100;
//
//    for (int iteration = 0; iteration < num_iterations; ++iteration) {
//        std::unordered_set<size_t> delete_indices_set;
//
//        // Calculate starting index for deleting a portion of the data
//        size_t start_idx = last_idx;
//        int num_to_delete = num_data / num_iterations;
//        last_idx = start_idx + num_to_delete;
//
//        // Add indices for deletion
//        for (size_t idx = start_idx; idx < start_idx + num_to_delete; ++idx) {
//            delete_indices_set.insert(idx);
//        }
//
//        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());
//
//        // Save deleted vectors
//        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
//        std::vector<faiss::idx_t> delete_ids(delete_indices.size());
//        for (size_t i = 0; i < delete_indices.size(); ++i) {
//            size_t idx = delete_indices[i];
//            deleted_vectors[i] = data[idx];
//            delete_ids[i] = static_cast<faiss::idx_t>(idx);  // Use original IDs
//        }
//
//        // Perform deletion using Faiss's remove_ids
//        faiss::IDSelectorArray selector(delete_ids.size(), delete_ids.data());
//        auto start_time_delete = std::chrono::high_resolution_clock::now();
//        index->remove_ids(selector);
//        auto end_time_delete = std::chrono::high_resolution_clock::now();
//        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();
//
//        // Flatten deleted_vectors to a 1D array for add_with_ids
//        std::vector<float> flat_deleted_vectors;
//        for (const auto& vec : deleted_vectors) {
//            flat_deleted_vectors.insert(flat_deleted_vectors.end(), vec.begin(), vec.end());
//        }
//
//        // Re-add the deleted vectors with their original IDs
//        auto start_time_add = std::chrono::high_resolution_clock::now();
//        index->add_with_ids(delete_ids.size(), flat_deleted_vectors.data(), delete_ids.data());
//        auto end_time_add = std::chrono::high_resolution_clock::now();
//        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();
//
//        // Perform k-NN search using Faiss search
//        std::vector<faiss::idx_t> I(num_queries * k);
//        std::vector<float> D(num_queries * k);
//
//        auto start_time_query = std::chrono::high_resolution_clock::now();
//        index->search(num_queries, queries[0].data(), k, D.data(), I.data());
//        auto end_time_query = std::chrono::high_resolution_clock::now();
//        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();
//
//        // Reshape Faiss search results for recall calculation
//        size_t total_recall_hits = 0;
//        for (size_t i = 0; i < num_queries; ++i) {
//            std::unordered_set<int> gt_set(ground_truth[i].begin(), ground_truth[i].begin() + k);
//            for (size_t j = 0; j < k; ++j) {
//                if (gt_set.find(I[i * k + j]) != gt_set.end()) {
//                    total_recall_hits++;
//                }
//            }
//        }
//
//        float recall = static_cast<float>(total_recall_hits) / (num_queries * k);
//
//        auto avg_delete_time = delete_duration / num_to_delete;
//        auto avg_add_time = add_duration / num_to_delete;
//        auto avg_query_time = query_duration / num_queries;
//
//        auto avg_sum_delete_add_time = avg_delete_time + avg_add_time;
//
//        std::cout << "------------------------------------------------------------------" << std::endl;
//        std::cout << "Iteration " << iteration + 1 << ":\n";
//        std::cout << "RECALL: " << recall << "\n";
//        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
//        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
//        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
//        std::cout << "Avg SUM Delete Add Time: " << avg_sum_delete_add_time << " seconds\n";
//
//        std::string iteration_string = std::to_string(iteration + 1);
//        std::string unreachable_points_string = "0";  // Always set to 0 as requested
//        std::string recall_string = std::to_string(recall);
//        std::string avg_delete_time_string = std::to_string(avg_delete_time);
//        std::string avg_add_time_string = std::to_string(avg_add_time);
//        std::string avg_sum_delete_add_time_string = std::to_string(avg_sum_delete_add_time);
//        std::string avg_query_time_string = std::to_string(avg_query_time);
//
//        std::vector<std::vector<std::string>> result_data = {{iteration_string, unreachable_points_string, recall_string, avg_delete_time_string, avg_add_time_string, avg_sum_delete_add_time_string, avg_query_time_string}};
//        util::writeCSVApp(output_csv_path, result_data);
//    }
//
//    faiss::write_index(index, output_index_path.c_str());
//
//    return 0;
//}
