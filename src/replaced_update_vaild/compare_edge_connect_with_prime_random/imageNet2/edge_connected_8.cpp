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

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/data/imageNet/image.ds";
    std::string query_path = root_path + "/data/imageNet/image.q";
    std::string index_path = root_path + "/data/imageNet/test_index/imageNet_hnsw_maintenance_index_1.bin";
    std::string ground_truth_path = root_path + "/data/imageNet/imageNet_groundtruth.ivecs";
    std::string random_indice_path = root_path + "/data/imageNet/imageNet_random_data.ivecs";
    std::string output_csv_path = root_path + "/output/random/imageNet2/edge_connected_8.csv";
    std::string output_index_path = root_path + "/output/random/imageNet2/edge_connected_8_imageNet_random_index.bin";

    std::vector<std::string> paths_to_create ={output_csv_path,output_index_path};
    util::create_directories(paths_to_create);



    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
    size_t data_siz = data.size();

    int random_data_num_per_iteration, num_iterations;
    std::vector<std::vector<size_t>> random_data = util::load_ivecs(random_indice_path,random_data_num_per_iteration,num_iterations);


    int k = 100;

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


    std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+5);
    auto results = util::query_index(&index, queries_tmp, data_siz);
    std::unordered_map <size_t,bool> excluded_global_labels_all;
    for (size_t j = 0; j < queries_tmp.size(); ++j) {
        std::cout << "Query " << j << ":" << std::endl;
        std::cout << "Labels length: " << results[j].first.size() << ",只能找到这么多的点" << std::endl;
    }


    std::vector<std::vector<size_t>> labels;
    auto start_time_query = std::chrono::high_resolution_clock::now();
    util::query_hnsw(index, queries, k, num_threads, labels);
    auto end_time_query = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

    auto avg_query_time = query_duration / queries.size();
    std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";

    // generate CSV file
    std::vector<std::vector <std::string>> header = {{"iteration_number" , "unreachable_points_number","recall","avg_delete_time",
                                                      "avg_add_time","avg_sum_delete_add_time","avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);


    size_t last_idx = 0;
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
//        double delete_rate = 0.01;
//        int num_to_delete = num_data * delete_rate;
//        std::unordered_set<size_t> delete_indices_set;
//        while (delete_indices_set.size() < num_to_delete) {
//            size_t idx = std::uniform_int_distribution<size_t>(0, num_data - 1)(gen);
//            delete_indices_set.insert(idx);
//        }
//
//        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());

        int num_to_delete = random_data[iteration].size();
        std::vector<size_t> delete_indices(random_data[iteration].begin(),random_data[iteration].end());

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