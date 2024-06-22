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

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string first_half_data_path = root_path + "/data/sift_2M/sift_2M_first_half.fvecs";
    std::string second_half_data_path = root_path + "/data/sift_2M/sift_2M_back_half.fvecs";

    std::string query_path = root_path + "/data/sift_2M/bigann_query.fvecs";

//    std::string index_path = root_path + "/data/sift_2M/hnsw_maintenance/sift2M_hnsw_maintenance_index.bin";

    std::string index_path = root_path + "/data/sift_2M/hnsw_prime/sift2M_hnsw_prime_index.bin";

    std::string ground_truth_base_path = root_path + "/data/sift_2M/groundTruth/sift_2M_";

    std::string output_csv_path = root_path + "/output/new_insert/sift_2M/edge_connected_replaced_update9.csv";
    std::string output_index_path = root_path + "/output/new_insert/sift_2M/edge_connected_replaced_update9_sift_2M_new_insert_index.bin";


    int dim, num_data_first_half, num_data_second_half, num_queries;
    std::vector<std::vector<float>> first_half_data = util::load_fvecs(first_half_data_path, dim, num_data_first_half);
    std::vector<std::vector<float>> second_half_data = util::load_fvecs(second_half_data_path, dim, num_data_second_half);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);

    size_t total_data_size = first_half_data.size() + second_half_data.size();
    int k = 100;

    // Initialize the HNSW index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, first_half_data.size(), true);
    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < total_data_size; ++i) {
        index_map[i] = i;
    }

    std::cout << "索引加载完毕 " << std::endl;
    // 设置查询参数`ef`
    int ef = 500;
    index.setEf(ef);

    int num_threads = 40;

    // Number of iterations for delete and re-add process
    int num_iterations = 10; // 因为每次删除和插入的数量是10万
    int num_to_delete_or_insert = 100000; // 每次删除和插入的数量

    // generate CSV file
    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time",
                                                     "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        std::unordered_set<size_t> delete_indices_set;
        std::unordered_set<size_t> insert_indices_set;

        // 删除前一百万个结点的前10万个结点
        size_t start_idx = iteration * num_to_delete_or_insert;
        size_t end_idx = start_idx + num_to_delete_or_insert;
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            delete_indices_set.insert(idx);
        }

        // 插入后一百万个结点的前10万个结点
        size_t insert_start_idx = iteration * num_to_delete_or_insert;
        size_t insert_end_idx = insert_start_idx + num_to_delete_or_insert;
        for (size_t idx = insert_start_idx; idx < insert_end_idx; ++idx) {
            insert_indices_set.insert(first_half_data.size()+ idx);
        }

        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());
        std::vector<size_t> insert_indices(insert_indices_set.begin(), insert_indices_set.end());

        // Save the vectors and their labels to be deleted before deleting them
        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = delete_indices[i];
            deleted_vectors[i] = first_half_data[idx];
        }
//        index.checkTotalInOutDegreeEquality();

        auto start_time_delete = std::chrono::high_resolution_clock::now();
        util::markDeleteMultiThread(index, delete_indices, index_map, num_threads);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

//        index.checkTotalInOutDegreeEquality();

        // Re-add the deleted vectors with their new labels

        std::vector<std::vector<float>> insert_vectors(insert_indices.size(), std::vector<float>(dim));
        for (size_t i = 0; i < insert_indices.size(); ++i) {
            size_t idx = insert_indices[i] - num_data_first_half;
            insert_vectors[i] = second_half_data[idx];
        }
        auto start_time_add = std::chrono::high_resolution_clock::now();
        util::addPointsMultiThread(index, insert_vectors, insert_indices, num_threads);
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        // Perform k-NN search and measure recall and query time
        std::vector<std::vector<size_t>> labels;

        auto start_time_query = std::chrono::high_resolution_clock::now();
        util::query_hnsw(index, queries, k, num_threads, labels);
        auto end_time_query = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

        // 计算RECALL
        std::string ground_truth_file = ground_truth_base_path + std::to_string(iteration + 1) + ".ivecs";
        std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_file);
        float recall = util::recall_score(ground_truth, labels, index_map, total_data_size);

        auto avg_delete_time = delete_duration / num_to_delete_or_insert;
        auto avg_add_time = add_duration / num_to_delete_or_insert;
        auto avg_query_time = query_duration / queries.size();
        auto avg_sum_delete_add_time = avg_delete_time + avg_add_time;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Iteration " << iteration + 1 << ":\n";
        std::cout << "删除了大约" << (double)num_to_delete_or_insert / first_half_data.size() << "的点" << std::endl;
        std::cout << "RECALL: " << recall << "\n";
        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Avg SUM Delete Add Time: " << avg_sum_delete_add_time << " seconds\n";

        std::vector<std::vector<float>> queries_tmp(queries.begin(), queries.begin() + 1);
        auto results = util::query_index(&index, queries_tmp, first_half_data.size());
        std::unordered_map<size_t, bool> excluded_global_labels_all;
        for (size_t j = 0; j < queries_tmp.size(); ++j) {
            std::cout << "Query " << j << ":" << std::endl;
            std::cout << "Labels length: " << results[j].first.size() << ",只能找到这么多的点" << std::endl;
        }
        std::cout << "------------------------------------------------------------------" << std::endl;

        std::string iteration_string = std::to_string(iteration + 1);
        std::string unreachable_points_string = std::to_string(first_half_data.size() - results.front().first.size());
        std::string recall_string = std::to_string(recall);
        std::string avg_delete_time_string = std::to_string(avg_delete_time);
        std::string avg_add_time_string = std::to_string(avg_add_time);
        std::string avg_sum_delete_add_time_string = std::to_string(avg_sum_delete_add_time);
        std::string avg_query_time_string = std::to_string(avg_query_time);

        std::vector<std::vector<std::string>> result_data = {{iteration_string, unreachable_points_string, recall_string,
                                                              avg_delete_time_string, avg_add_time_string, avg_sum_delete_add_time_string, avg_query_time_string}};

        util::writeCSVApp(output_csv_path, result_data);
    }
    index.saveIndex(output_index_path);
    return 0;
}
