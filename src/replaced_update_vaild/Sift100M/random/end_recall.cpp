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


// 添加 load_bvecs_range 函数
std::vector<std::vector<float>> load_bvecs_range(const std::string& filename, size_t start_idx, size_t count, int dim) {
    std::vector<std::vector<float>> data;
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    size_t vector_size_in_bytes = sizeof(int) + sizeof(uint8_t) * dim;
    size_t start_pos = start_idx * vector_size_in_bytes;

    // 移动文件指针到起始位置
    input.seekg(start_pos, std::ios::beg);
    if (input.fail()) {
        std::cerr << "无法定位到文件中的起始位置。" << std::endl;
        exit(1);
    }

    for (size_t i = 0; i < count; ++i) {
        int cur_dim = 0;
        input.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (cur_dim != dim) {
            std::cerr << "维度不匹配，预期: " << dim << ", 实际: " << cur_dim << std::endl;
            exit(1);
        }
        std::vector<uint8_t> vec_uint8(dim);
        input.read(reinterpret_cast<char*>(vec_uint8.data()), sizeof(uint8_t) * dim);
        if (input.fail()) {
            std::cerr << "读取向量失败。" << std::endl;
            exit(1);
        }
        // 将 uint8_t 数据转换为 float
        std::vector<float> vec_float(dim);
        for (int d = 0; d < dim; ++d) {
            vec_float[d] = static_cast<float>(vec_uint8[d]);
        }
        data.push_back(std::move(vec_float));
    }

    input.close();
    return data;
}

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }

    std::string root_path = argv[1]; //  /disk2/xiaowentao


    std::vector <std::vector <std::string>> csv_index_path_vec={
        {root_path + "/output/random/sift200M/edge_connected_replaced_update7_end_recall.csv",
         root_path + "/output/random/sift200M/edge_connected_replaced_update7_sift100M_random_index.bin",
         root_path + "/sift/bigann_base.bvecs",
         root_path + "/sift/sift200M/bigann_query.bvecs",
         root_path + "/sift/sift200M/gnd/idx_100M.ivecs",},
        {root_path + "/output/random/sift200M/edge_connected_replaced_update8_end_recall.csv",
                root_path + "/output/random/sift200M/edge_connected_replaced_update8_sift100M_random_index.bin",
                root_path + "/sift/bigann_base.bvecs",
                root_path + "/sift/sift200M/bigann_query.bvecs",
                root_path + "/sift/sift200M/gnd/idx_100M.ivecs",},
        {root_path + "/output/random/sift200M/edge_connected_replaced_update9_end_recall.csv",
         root_path + "/output/random/sift200M/edge_connected_replaced_update9_sift100M_random_index.bin",
         root_path + "/sift/bigann_base.bvecs",
         root_path + "/sift/sift200M/bigann_query.bvecs",
         root_path + "/sift/sift200M/gnd/idx_100M.ivecs",},
            {root_path + "/output/random/sift200M/edge_connected_replaced_update10_end_recall.csv",
             root_path + "/output/random/sift200M/edge_connected_replaced_update10_sift100M_random_index.bin",
             root_path + "/sift/bigann_base.bvecs",
             root_path + "/sift/sift200M/bigann_query.bvecs",
             root_path + "/sift/sift200M/gnd/idx_100M.ivecs",},
            {root_path + "/output/random/sift200M/replaced_update_end_recall.csv",
             root_path + "/output/random/sift200M/replaced_update_sift100M_random_index.bin",
             root_path + "/sift/bigann_base.bvecs",
             root_path + "/sift/sift200M/bigann_query.bvecs",
             root_path + "/sift/sift200M/gnd/idx_100M.ivecs",
             },




    };

    for(auto csv_index : csv_index_path_vec){
        auto csv_path = csv_index[0];
        auto index_path = csv_index[1];
//        auto data_path = csv_index[2];
        auto query_path = csv_index[3];
        auto ground_truth_path = csv_index[4];

        int dim = 128, num_queries = 10000;
//        std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
        std::vector<std::vector<float>> queries = load_bvecs_range(query_path, 0, num_queries, dim);
        size_t data_siz = 100000000;
        int k = 1000;
        if(data_siz == 2340373){
            k = 10;
        }
        std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
        std::unordered_map<size_t, size_t> index_map;
        for (size_t i = 0; i < data_siz; ++i) {
            index_map[i + data_siz] = i;
        }

        std::cout<<"开始加载索引"<<std::endl;

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


        std::cout<<"开始计算"<<std::endl;
        std::vector<std::vector<double>> recall_time_vector = util::countRecallWithDiffPara(index,queries,ground_truth,index_map,k,start_ef,end_ef,step,num_threads,data_siz);
        std::cout<<"计算完毕"<<std::endl;

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