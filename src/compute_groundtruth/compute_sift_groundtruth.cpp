// main.cpp
#include "util.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string first_half_file = root_path + "/data/sift_2M/sift_2M_first_half.fvecs";
    std::string second_half_file = root_path + "/data/sift_2M/sift_2M_back_half.fvecs";
    std::string query_file = root_path + "/data/sift_2M/bigann_query.fvecs";
    std::string output_dir = root_path + "/data/sift_2M/groundTruth";



    int dim, num_data, num_queries;
    std::vector<std::vector<float>> first_half_vectors = util::load_fvecs(first_half_file, dim, num_data);
    std::vector<std::vector<float>> second_half_vectors = util::load_fvecs(second_half_file, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_file, dim, num_queries);

    int step_size = 100000; // 每轮前移10万
    int num_iterations = 10; // 一共10轮
    int window_size = 1000000; // 每次保留的总向量数量

    // Ensure output directory exists
    std::system(("mkdir -p " + output_dir).c_str());

//    queries.resize(10);

    for (int i = 0; i <= num_iterations; ++i) {
        int start_idx = i * step_size;
//        int first_half_end_idx = start_idx + window_size - step_size;

        std::vector<std::vector<float>> combined_vectors;
        combined_vectors.insert(combined_vectors.end(), first_half_vectors.begin() + start_idx, first_half_vectors.end());
        combined_vectors.insert(combined_vectors.end(), second_half_vectors.begin(), second_half_vectors.begin() + start_idx);

        auto knn_results = util::brute_force_knn(combined_vectors, queries, dim, 100);

        // Adjust indices to the global range (0 to 2000000)
        for (auto& knn : knn_results) {
            for (auto& idx : knn) {
                idx += start_idx;
//                if (idx < window_size - step_size) {
//                    idx += start_idx; // Adjust for first half vectors
//                } else {
//                    idx = idx - (window_size - step_size) + num_data; // Adjust for second half vectors
//                }
            }
        }

        std::string output_file = output_dir + "/sift_2M_" + std::to_string(i) + ".ivecs";
        util::save_knn_to_ivecs(output_file, knn_results);
        std::cout << "Iteration " << i << ": Saved ground truth to " << output_file << std::endl;
    }

    return 0;
}
