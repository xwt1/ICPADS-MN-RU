//
// Created by root on 6/7/24.
//

#include "util.h"
#include "iostream"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/data/yahoomusic/yahoomusic_base.fvecs";
    std::string query_data_path = root_path + "/data/yahoomusic/yahoomusic_query_all.fvecs";
    std::string ground_truth_path = root_path + "/data/yahoomusic/yahoomusic_groundtruth.ivecs";

//    std::string data_path = root_path + "/data/gist/gist_base.fvecs";
//    std::string query_data_path = root_path + "/data/gist/gist_query.fvecs";
//    std::string ground_truth_path = root_path + "/data/gist/gist_groundtruth_myself.ivecs";



    int dim,k=5,num_data,num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_data_path, dim, num_queries);

    auto knn_results = util::brute_force_knn(data, queries, dim, k);
    util::save_knn_to_ivecs(ground_truth_path, knn_results);

    return 0;
}