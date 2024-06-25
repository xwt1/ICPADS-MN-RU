//
// Created by root on 6/21/24.
//

//
// Created by root on 6/7/24.
//

#include "util.h"
#include "iostream"


int main() {
    std::string root_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement";

    std::string data_path = root_path + "/data/imageNet/image.ds";
    std::string ground_truth_path = root_path + "/data/imageNet/imageNet_2M_groundtruth.ivecs";

    // only top-1
    int dim,k=1,num_data,num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(data_path, dim, num_queries);

    std::vector<std::vector<size_t>> knn_results;
    for(size_t i = 0; i < num_data; i++){
        std::vector <size_t> query_result = {i};
        knn_results.push_back(query_result);
    }
//    auto knn_results = util::brute_force_knn(data, queries, dim, k);
    util::save_knn_to_ivecs(ground_truth_path, knn_results);
    return 0;
}