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
//    std::string data_path = root_path + "/data/gist/gist_base.fvecs";
//    std::string query_path = root_path + "/data/gist/gist_query.fvecs";
//    std::string ground_truth_path = root_path + "/data/gist/gist_groundtruth.ivecs";



    std::vector <std::vector <std::string>> csv_index_path_vec={
        {root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update7_end_recall.csv",
         root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update7_word2vec_full_coverage_index.bin",
             root_path + "/data/word2vec/word2vec_base.fvecs",
             root_path + "/data/word2vec/word2vec_query.fvecs",
             root_path + "/data/word2vec/word2vec_groundtruth.ivecs"},
        {root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update8_end_recall.csv",
                root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update8_word2vec_full_coverage_index.bin",
                root_path + "/data/word2vec/word2vec_base.fvecs",
                root_path + "/data/word2vec/word2vec_query.fvecs",
                root_path + "/data/word2vec/word2vec_groundtruth.ivecs"},
        {root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update9_end_recall.csv",
                root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update9_word2vec_full_coverage_index.bin",
                root_path + "/data/word2vec/word2vec_base.fvecs",
                root_path + "/data/word2vec/word2vec_query.fvecs",
                root_path + "/data/word2vec/word2vec_groundtruth.ivecs"},
        {root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update10_end_recall.csv",
                root_path + "/output/full_coverage/word2vec/edge_connected_replaced_update10_word2vec_full_coverage_index.bin",
                root_path + "/data/word2vec/word2vec_base.fvecs",
                root_path + "/data/word2vec/word2vec_query.fvecs",
                root_path + "/data/word2vec/word2vec_groundtruth.ivecs"},
        {root_path + "/output/full_coverage/word2vec/replaced_update_end_recall.csv",
         root_path + "/output/full_coverage/word2vec/replaced_update_word2vec_full_coverage_index.bin",
                root_path + "/data/word2vec/word2vec_base.fvecs",
                root_path + "/data/word2vec/word2vec_query.fvecs",
                root_path + "/data/word2vec/word2vec_groundtruth.ivecs"},
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
        int k = 100;
        if(data_siz == 2340373){
            k = 10;
        }
        std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
        std::unordered_map<size_t, size_t> index_map;
        for (size_t i = 0; i < num_data; ++i) {
            index_map[i + data_siz] = i;
        }


        // Initialize the HNSW index
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
        std::cout << "索引加载完毕 " << std::endl;

        int start_ef = 1000;
        int end_ef = 10000;
        int step = 100;
        int num_threads = 40;

//        if(data_siz == 2340373){
//            start_ef =600;
//            end_ef = 10000;
//        }



        std::vector<std::vector<double>> recall_time_vector = util::countRecallWithDiffPara(index,queries,ground_truth,index_map,k,start_ef,end_ef,step,num_threads,data_siz);

        // generate CSV file
        std::vector<std::vector <std::string>> header = {{"ef" , "recall", "query_time"}};
        util::writeCSVOut(csv_path, header);

        for(auto recall_time : recall_time_vector){

            std::string ef_string = std::to_string(start_ef);
            std::string recall_string = std::to_string(recall_time[0]);
            std::string query_time_string = std::to_string(recall_time[1]);
            std::vector<std::vector <std::string>> result_data = {{ef_string, recall_string,query_time_string}};
            util::writeCSVApp(csv_path, result_data);
            start_ef += step;
        }
    }

    return 0;
}