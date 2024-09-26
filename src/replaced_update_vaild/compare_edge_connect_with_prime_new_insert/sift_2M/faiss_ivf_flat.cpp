#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <unordered_set>
#include <chrono>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
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

    std::string index_path = root_path + "/data/sift/faiss_index/sift_index.bin";

    std::string ground_truth_base_path = root_path + "/data/sift_2M/groundTruth/sift_2M_";

    std::string output_csv_path = root_path + "/output/new_insert/sift_2M/faiss_ivf_flat.csv";
    std::string output_index_path = root_path + "/output/new_insert/sift_2M/faiss_sift_output_index.bin";

    int dim, num_data_first_half, num_data_second_half, num_queries;
    std::vector<std::vector<float>> first_half_data = util::load_fvecs(first_half_data_path, dim, num_data_first_half);
    std::vector<std::vector<float>> second_half_data = util::load_fvecs(second_half_data_path, dim, num_data_second_half);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);

    size_t total_data_size = first_half_data.size() + second_half_data.size();
    int k = 100;

    // Load the existing Faiss IVF-FLAT index from the specified path
    faiss::IndexIVFFlat* index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(index_path.c_str()));
    if (!index) {
        std::cerr << "Failed to load the Faiss index from " << index_path << std::endl;
        return 1;
    }

    std::cout << "Faiss IVF-FLAT 索引加载完毕 " << std::endl;
    int nprobe = 10;  // Set the search parameter nprobe
    index->nprobe = nprobe;

    int num_threads = 40;
    int num_iterations = 10; // Iteration count
    int num_to_delete_or_insert = 100000; // Number of vectors to delete/insert per iteration

    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time",
                                                     "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);

    // Keep track of deleted IDs
    std::unordered_set<faiss::idx_t> deleted_ids;

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        std::unordered_set<size_t> delete_indices_set;
        std::unordered_set<size_t> insert_indices_set;

        size_t start_idx = iteration * num_to_delete_or_insert;
        size_t end_idx = start_idx + num_to_delete_or_insert;
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            delete_indices_set.insert(idx);
        }

        size_t insert_start_idx = iteration * num_to_delete_or_insert;
        size_t insert_end_idx = insert_start_idx + num_to_delete_or_insert;
        for (size_t idx = insert_start_idx; idx < insert_end_idx; ++idx) {
            insert_indices_set.insert(first_half_data.size() + idx);
        }

        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());
        std::vector<size_t> insert_indices(insert_indices_set.begin(), insert_indices_set.end());

        // Mark IDs as deleted
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        for (size_t idx : delete_indices) {
            deleted_ids.insert(static_cast<faiss::idx_t>(idx));
        }
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

        // Prepare vectors to be inserted in a flat array for batch insertion
        std::vector<float> insert_data(dim * insert_indices.size());
        for (size_t i = 0; i < insert_indices.size(); ++i) {
            size_t idx = insert_indices[i] - num_data_first_half;
            std::copy(second_half_data[idx].begin(), second_half_data[idx].end(), insert_data.begin() + i * dim);
        }

        auto start_time_add = std::chrono::high_resolution_clock::now();
        index->add(insert_indices.size(), insert_data.data());
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        // Perform k-NN search using Faiss
        std::vector<faiss::idx_t> I(num_queries * k);
        std::vector<float> D(num_queries * k);

        std::vector<float> flat_queries(num_queries * dim);
        for (size_t i = 0; i < num_queries; ++i) {
            std::copy(queries[i].begin(), queries[i].end(), flat_queries.begin() + i * dim);
        }

        auto start_time_query = std::chrono::high_resolution_clock::now();
        index->search(num_queries, flat_queries.data(), k, D.data(), I.data());
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
        std::string ground_truth_file = ground_truth_base_path + std::to_string(iteration + 1) + ".ivecs";
        std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_file);

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

        auto avg_delete_time = delete_duration / num_to_delete_or_insert;
        auto avg_add_time = add_duration / num_to_delete_or_insert;
        auto avg_query_time = query_duration / queries.size();
        auto avg_sum_delete_add_time = avg_delete_time + avg_add_time;

        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "Iteration " << iteration + 1 << ":\n";
        std::cout << "RECALL: " << recall << "\n";
        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Avg SUM Delete Add Time: " << avg_sum_delete_add_time << " seconds\n";

        std::string iteration_string = std::to_string(iteration + 1);
        std::string unreachable_points_string = "0";
        std::string recall_string = std::to_string(recall);
        std::string avg_delete_time_string = std::to_string(avg_delete_time);
        std::string avg_add_time_string = std::to_string(avg_add_time);
        std::string avg_sum_delete_add_time_string = std::to_string(avg_sum_delete_add_time);
        std::string avg_query_time_string = std::to_string(avg_query_time);

        std::vector<std::vector<std::string>> result_data = {{iteration_string, unreachable_points_string, recall_string,
                                                              avg_delete_time_string, avg_add_time_string, avg_sum_delete_add_time_string, avg_query_time_string}};

        util::writeCSVApp(output_csv_path, result_data);
    }
    faiss::write_index(index, output_index_path.c_str());
    delete index;  // Free the index
    return 0;
}
