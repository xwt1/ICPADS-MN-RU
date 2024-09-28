#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <unordered_set>
#include <chrono>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include "util.h"
#include <omp.h>  // Include OpenMP header

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string first_half_data_path = root_path + "/data/sift_2M/sift_2M_first_half.fvecs";
    std::string second_half_data_path = root_path + "/data/sift_2M/sift_2M_back_half.fvecs";

    std::string query_path = root_path + "/data/sift_2M/bigann_query.fvecs";

    std::string index_path = root_path + "/data/sift_2M/faiss_index/sift2M_index.bin";

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
    int nprobe = 200;  // Set the search parameter nprobe
    index->nprobe = nprobe;

    int num_threads = 40;
    omp_set_num_threads(num_threads);

    int num_iterations = 10; // Iteration count
    int num_to_delete_or_insert = 100000; // Number of vectors to delete/insert per iteration

    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time",
                                                     "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        // Define the deletion and insertion range for this iteration
        faiss::idx_t start_id = iteration * num_to_delete_or_insert;
        faiss::idx_t end_id = start_id + num_to_delete_or_insert;

        // Prepare the vector of IDs to be deleted
        std::vector<faiss::idx_t> ids_to_delete;
        for (faiss::idx_t idx = start_id; idx < end_id; ++idx) {
            ids_to_delete.push_back(idx);
        }

        // Remove the IDs using remove_ids
        faiss::IDSelectorArray id_selector(ids_to_delete.size(), ids_to_delete.data());
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        index->remove_ids(id_selector);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

        std::cout<<index->ntotal<<std::endl;

        // Prepare vectors to be inserted in a flat array for batch insertion with IDs
        std::vector<float> insert_data(dim * num_to_delete_or_insert);
        std::vector<faiss::idx_t> insert_ids(num_to_delete_or_insert);

        for (faiss::idx_t i = 0; i < num_to_delete_or_insert; ++i) {
            size_t idx = i + start_id;
            insert_ids[i] = idx + num_data_first_half;  // ID shift
            std::copy(second_half_data[idx].begin(), second_half_data[idx].end(), insert_data.begin() + i * dim);
        }

        // Use add_with_ids to add vectors back with their specific IDs
        auto start_time_add = std::chrono::high_resolution_clock::now();
        index->add_with_ids(num_to_delete_or_insert, insert_data.data(), insert_ids.data());
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        std::cout<<index->ntotal<<std::endl;

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

        std::vector<std::vector<std::string>> result_data = {{std::to_string(iteration + 1), "0", std::to_string(recall),
                                                              std::to_string(avg_delete_time), std::to_string(avg_add_time),
                                                              std::to_string(avg_sum_delete_add_time), std::to_string(avg_query_time)}};

        util::writeCSVApp(output_csv_path, result_data);
    }

    faiss::write_index(index, output_index_path.c_str());
    delete index;  // Free the index
    return 0;
}
