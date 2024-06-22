//
// Created by root on 6/10/24.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <hnswlib/hnswlib.h>
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include "util.h"


void output_CSV(std::string index_path,
                std::string query_data_path,
                std::string data_path,
                std::string dataset_name,
                std::string output_csv_path,
                std::string groundTruth_path,
                int ef = 1000000){


}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/data/sift/sift_base.fvecs";
    std::string index_path = root_path + "/data/sift/hnsw_prime/sift_hnsw_prime_index.bin";
    std::string query_data_path = root_path + "/data/sift/sift_query.fvecs";

    std::string ground_truth_path = root_path + "/data/sift/sift_groundtruth.ivecs";
    std::string output_csv_path = root_path + "/output/figure_2/unreachable_points_phenomenon.csv";

    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_data_path, dim, num_queries);

    size_t data_siz = data.size();

    int k = 100;
    int num_threads = std::thread::hardware_concurrency();
//    int num_threads = 1;

    // Initialize the HNSW index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    std::cout << "索引加载完毕 " << std::endl;
    // 设置查询参数`ef`
    int ef = 100;
    index.setEf(ef);

    // Number of iterations for delete and re-add process
    int num_iterations = 500;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Perform initial brute-force k-NN search to get ground truth
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);


    // generate CSV file
    std::vector<std::vector <std::string>> header = {{"iteration_number" , "unreachable_points_number"}};
    util::writeCSVOut(output_csv_path,header);

    std::cout<<"------------------------------------------------------------------"<<std::endl;
    std::cout << "Iteration " << 0 << ":\n";
    std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+5);
    index.setEf(1000000);
    auto results = util::query_index(&index, queries_tmp, 1000000);
    for (size_t j = 0; j < queries_tmp.size(); ++j) {
        std::cout << "Query " << j << ":" << std::endl;
        std::cout << "Only find " << results[j].first.size() << " points" << std::endl;
    }
    index.setEf(ef);
    std::string iteration_number = std::to_string(0);
    std::string unreachable_points_number = std::to_string(data_siz - results.front().first.size());
    std::vector<std::vector <std::string>> result_data = {{iteration_number, unreachable_points_number}};
    util::writeCSVApp(output_csv_path,result_data);
    std::cout<<"------------------------------------------------------------------"<<std::endl;

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        // Select the indices to delete
        int num_to_delete = num_data * 0.05;
        std::unordered_set<size_t> delete_indices_set;
        while (delete_indices_set.size() < num_to_delete) {
            size_t idx = std::uniform_int_distribution<size_t>(0, num_data - 1)(gen);
            delete_indices_set.insert(idx);
        }
        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());

        // Save the vectors and their labels to be deleted before deleting them
        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = delete_indices[i];
            deleted_vectors[i] = data[idx];
        }

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
        util::query_hnsw(index, queries, k,num_threads, labels);
        auto end_time_query = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

        float recall = util::recall_score(ground_truth, labels, index_map, data_siz);
        double avg_query_time = query_duration / queries.size();


        std::cout<<"------------------------------------------------------------------"<<std::endl;
        std::cout << "Iteration " << iteration + 1 << ":\n";
        std::cout << "RECALL: " << recall << "\n";
        std::cout << "avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "avg Delete Time: " << delete_duration / num_to_delete<< " seconds\n";
        std::cout << "avg Add Time: " << add_duration / num_to_delete<< " seconds\n";

//        std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+5);
        index.setEf(1000000);
        results = util::query_index(&index, queries_tmp, 1000000);
        for (size_t j = 0; j < queries_tmp.size(); ++j) {
            std::cout << "Query " << j << ":" << std::endl;
            std::cout << "Only find " << results[j].first.size() << " points" << std::endl;
        }
        index.setEf(ef);
        std::cout<<"------------------------------------------------------------------"<<std::endl;
        iteration_number = std::to_string(iteration);
        unreachable_points_number = std::to_string(data_siz - results.front().first.size());
        result_data = {{iteration_number, unreachable_points_number}};
        util::writeCSVApp(output_csv_path,result_data);

    }

    return 0;
}