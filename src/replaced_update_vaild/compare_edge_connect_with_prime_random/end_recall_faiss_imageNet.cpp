//
// Created by root on 6/22/24.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include "faiss/AutoTune.h"
#include "util.h"
// 添加这行以启用多线程支持
#include <omp.h> // OpenMP for controlling thread count

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }

    std::string root_path = argv[1];

    std::vector <std::vector <std::string>> csv_index_path_vec = {
            {root_path + "/output/random/imageNet/faiss_end_recall_imageNet_2M.csv",
             root_path + "/output/random/imageNet/faiss_imageNet_output_index.bin",
             root_path + "/data/imageNet/image.ds",
             root_path + "/data/imageNet/image.q",
             root_path + "/data/imageNet/imageNet_groundtruth.ivecs"},
    };

    for(auto csv_index : csv_index_path_vec){
        auto csv_path = csv_index[0];
        auto index_path = csv_index[1];
        auto data_path = csv_index[2];
        auto query_path = csv_index[3];
        auto ground_truth_path = csv_index[4];

        int dim, num_data, num_queries;
        std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
        std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
        size_t data_siz = data.size();
        int k = 10;

        std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);

        // Flatten data vectors to match Faiss's requirements
        std::vector<float> flat_data(data_siz * dim);
        for (size_t i = 0; i < data_siz; ++i) {
            std::copy(data[i].begin(), data[i].end(), flat_data.begin() + i * dim);
        }

        std::vector<float> flat_queries(num_queries * dim);
        for (size_t i = 0; i < num_queries; ++i) {
            std::copy(queries[i].begin(), queries[i].end(), flat_queries.begin() + i * dim);
        }

        std::cout<<queries.size()<<std::endl;

        // Load the Faiss index
        faiss::Index* index = faiss::read_index(index_path.c_str());
        auto ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(index);
        if (!ivf_index) {
            std::cerr << "Error: Unable to load IVF-FLAT index from " << index_path << std::endl;
            return 1;
        }
        std::cout << "Faiss index loaded successfully." << std::endl;

        // Set parameters for searching
        int start_ef = 50;
        int end_ef = 150;
        int step = 5;
        int num_threads = 40;

        omp_set_num_threads(num_threads);
        std::vector<std::vector<double>> recall_time_vector;

        for (int ef = start_ef; ef <= end_ef; ef += step) {
            ivf_index->nprobe = ef; // Set the nprobe parameter

            auto start = std::chrono::high_resolution_clock::now();

            // Search for k nearest neighbors
            std::vector<float> distances(num_queries * k);
            std::vector<faiss::idx_t> indices(num_queries * k);

            index->search(num_queries, flat_queries.data(), k, distances.data(), indices.data());

            auto end = std::chrono::high_resolution_clock::now();
            double query_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
            double avg_query_time = query_time / queries.size(); // 计算平均查询时间

            // Calculate recall
            size_t correct = 0;
            for (size_t i = 0; i < num_queries; ++i) {
                std::unordered_set<size_t> gt_set(ground_truth[i].begin(), ground_truth[i].begin() + k);
                for (int j = 0; j < k; ++j) {
                    if (gt_set.count(indices[i * k + j])) {
                        correct++;
                    }
                }
            }

            double recall = static_cast<double>(correct) / (num_queries * k);
            recall_time_vector.push_back({recall, avg_query_time}); // 只保存 recall 和 avg_query_time

            std::cout << "nprobe: " << ef << ", recall: " << recall << ", avg_query_time: " << avg_query_time << "s" << std::endl;
        }

        // Save the results to a CSV file
        std::vector<std::vector<std::string>> header = {{"nprobe", "recall", "query_time"}}; // 只保留这三列
        util::writeCSVOut(csv_path, header);

        int current_ef = start_ef;
        for (const auto& recall_time : recall_time_vector) {
            std::vector<std::vector<std::string>> result_data = {
                    {std::to_string(current_ef), std::to_string(recall_time[0]),
                     std::to_string(recall_time[1])}}; // 保存 recall 和 avg_query_time
            util::writeCSVApp(csv_path, result_data);
            current_ef += step;
        }

        delete index; // Clean up the index
    }

    return 0;
}
