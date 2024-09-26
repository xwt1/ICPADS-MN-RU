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
#include <chrono>
#include <omp.h>  // Include OpenMP header
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include "faiss/AutoTune.h"
#include "faiss/IndexIDMap.h"
#include "faiss/utils/random.h"
#include "util.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/data/word2vec/word2vec_base.fvecs";
    std::string query_path = root_path + "/data/word2vec/word2vec_query.fvecs";
    std::string index_path = root_path + "/data/word2vec/faiss/word2vec_index.bin";
    std::string ground_truth_path = root_path + "/data/word2vec/word2vec_groundtruth.ivecs";
    std::string output_csv_path = root_path + "/output/full_coverage/word2vec/faiss_ivf_flat.csv";
    std::string output_index_path = root_path + "/output/full_coverage/word2vec/faiss_word2vec_output_index.bin";

    std::vector<std::string> paths_to_create = {output_csv_path, output_index_path};
    util::create_directories(paths_to_create);

    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);

    // 假设原来的二维数组是queries[num_queries][dim]
    std::vector<float> queries_1d(num_queries * dim);
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            queries_1d[i * dim + j] = queries[i][j];
        }
    }


    int k = 100;
    int num_threads = 40;  // Set your desired number of threads

    // Set the number of threads for Faiss operations
    omp_set_num_threads(num_threads);

    // Load Faiss IVF-Flat index
    faiss::IndexIVFFlat* index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(index_path.c_str()));

    if (!index) {
        std::cerr << "Failed to load Faiss index" << std::endl;
        return 1;
    }

    std::cout << index->ntotal << std::endl;

    index->nprobe = 100; // Set number of probes

    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    std::cout << "Index loaded successfully." << std::endl;



    int num_iterations = 100;

//    std::cout<<"wtf: "<<num_queries<<std::endl;
    // Load ground truth
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);

//// Perform an initial k-NN search using Faiss search to calculate initial recall
//// Perform an initial k-NN search using Faiss search to calculate initial recall
//    std::vector<faiss::idx_t> I(num_queries * k);
//    std::vector<float> D(num_queries * k);
//
//// 使用所有查询向量执行搜索
//    auto start_time_query = std::chrono::high_resolution_clock::now();
//    index->search(num_queries, queries_1d.data(), k, D.data(), I.data());
//    auto end_time_query = std::chrono::high_resolution_clock::now();
//    auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();
//
//// Reshape Faiss search results for recall calculation
//    std::vector<std::vector<size_t>> predictions(num_queries, std::vector<size_t>(k));
//    for (size_t i = 0; i < num_queries; ++i) {
//        std::cout<<"wtf2: "<<predictions[i].size()<<std::endl;
//        for (size_t j = 0; j < k; ++j) {
////            std::cout<<I[i * k + j]<<" ";
//            predictions[i][j] = I[i * k + j];
//            std::cout<<predictions[i][j]<<" ";
//        }
//        std::cout<<std::endl;
//    }
//
//// Calculate recall manually
//    size_t total_recall_hits = 0;
//    size_t total_possible = num_queries * k;
//
//    for (size_t i = 0; i < num_queries; ++i) {
//        const auto& gt = ground_truth[i];
//        const auto& pred = predictions[i];
//        std::unordered_set<size_t> gt_set(gt.begin(), gt.end());
//
//        for (size_t j = 0; j < k; ++j) {
//            if (gt_set.find(pred[j]) != gt_set.end()) {
//                total_recall_hits++;
//            }
//        }
//    }
//
//    float initial_recall = static_cast<float>(total_recall_hits) / total_possible;
//    std::cout << "Initial Recall: " << initial_recall << std::endl;



    // Generate CSV file header
    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time", "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);

    size_t last_idx = 0;
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        std::unordered_set<size_t> delete_indices_set;

        // Calculate starting index for deleting a portion of the data
        size_t start_idx = last_idx;
        int num_to_delete = num_data / num_iterations;
        last_idx = start_idx + num_to_delete;

        // Add indices for deletion
        for (size_t idx = start_idx; idx < start_idx + num_to_delete; ++idx) {
            delete_indices_set.insert(idx);
        }

        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());

        // Save deleted vectors
        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
        std::vector<faiss::idx_t> delete_ids(delete_indices.size());
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = delete_indices[i];
            deleted_vectors[i] = data[idx];
            delete_ids[i] = static_cast<faiss::idx_t>(idx);  // Use original IDs
        }

        // Perform deletion using Faiss's remove_ids
        faiss::IDSelectorArray selector(delete_ids.size(), delete_ids.data());
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        index->remove_ids(selector);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

        // Flatten deleted_vectors to a 1D array for add_with_ids
        std::vector<float> flat_deleted_vectors;
        for (const auto& vec : deleted_vectors) {
            flat_deleted_vectors.insert(flat_deleted_vectors.end(), vec.begin(), vec.end());
        }

        // Re-add the deleted vectors with their original IDs
        auto start_time_add = std::chrono::high_resolution_clock::now();
        index->add_with_ids(delete_ids.size(), flat_deleted_vectors.data(), delete_ids.data());
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        // Perform k-NN search using Faiss search
        std::vector<faiss::idx_t> I(num_queries * k);
        std::vector<float> D(num_queries * k);

        auto start_time_query = std::chrono::high_resolution_clock::now();
        index->search(num_queries, queries_1d.data(), k, D.data(), I.data());
        auto end_time_query = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

        // Reshape Faiss search results for recall calculation
        std::vector<std::vector<size_t>> predictions(num_queries, std::vector<size_t>(k));
        for (size_t i = 0; i < num_queries; ++i) {
            for (size_t j = 0; j < k; ++j) {
                predictions[i][j] = static_cast<size_t>(I[i * k + j]);
            }
        }

        // Calculate recall manually
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

        auto avg_delete_time = delete_duration / num_to_delete;
        auto avg_add_time = add_duration / num_to_delete;
        auto avg_query_time = query_duration / num_queries;

        auto avg_sum_delete_add_time = avg_delete_time + avg_add_time;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Iteration " << iteration + 1 << ":\n";
        std::cout << "RECALL: " << recall << "\n";
        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Avg SUM Delete Add Time: " << avg_sum_delete_add_time << " seconds\n";

        std::string iteration_string = std::to_string(iteration + 1);
        std::string unreachable_points_string = "0";  // Always set to 0 as requested
        std::string recall_string = std::to_string(recall);
        std::string avg_delete_time_string = std::to_string(avg_delete_time);
        std::string avg_add_time_string = std::to_string(avg_add_time);
        std::string avg_sum_delete_add_time_string = std::to_string(avg_sum_delete_add_time);
        std::string avg_query_time_string = std::to_string(avg_query_time);

        std::vector<std::vector<std::string>> result_data = {{iteration_string, unreachable_points_string, recall_string, avg_delete_time_string, avg_add_time_string, avg_sum_delete_add_time_string, avg_query_time_string}};
        util::writeCSVApp(output_csv_path, result_data);
    }

    faiss::write_index(index, output_index_path.c_str());

    return 0;
}
