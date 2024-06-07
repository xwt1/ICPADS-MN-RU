//
// Created by root on 6/6/24.
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

/*
 * CSV format: dataset name , query_time , delete_update_time
 */
void output_CSV(std::string index_path,
                std::string query_data_path,
                std::string data_path,
                std::string dataset_name,
                std::string output_csv_path,
                std::string groundTruth_path,
                int ef = 500){
    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_data_path, dim, num_queries);
    size_t data_siz = data.size();
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(groundTruth_path);

    int k = ground_truth[0].size();

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    index.setEf(ef);

    size_t last_idx = 0;
    {
        std::unordered_set<size_t> delete_indices_set;

        // set 10000 delete update point,
        size_t start_idx = last_idx;
        int num_to_delete = 1000;
        std::cout<<"delete "<<num_to_delete<<" points"<<std::endl;
        // 将后一半结点的下标插入到集合中
        for (size_t idx = start_idx; idx < start_idx+num_to_delete; ++idx) {
            delete_indices_set.insert(idx);
        }
        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());
        // Save the vectors and their labels to be deleted before deleting them
        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = delete_indices[i];
            deleted_vectors[i] = data[idx];
        }
        std::cout<<"begin delete"<<std::endl;
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        util::MarkDeleteMultiThread(index, delete_indices, index_map, 1);
        std::cout<<"delete complete"<<std::endl;
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();
        auto avg_delete_time = (double) delete_duration / num_to_delete;

        // Re-add the deleted vectors with their original labels
        std::vector<size_t> new_indices(delete_indices.size());
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = index_map[delete_indices[i]];
            size_t new_idx = (idx < num_data) ? idx + num_data : idx - num_data;
            new_indices[i] = new_idx;
            index_map[delete_indices[i]] = new_idx;
        }

        std::cout<<"begin re-add"<<std::endl;
        auto start_time_add = std::chrono::high_resolution_clock::now();
        util::addPointsMultiThread(index, deleted_vectors, new_indices, 1);
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();
        double avg_add_time = (double) add_duration / num_to_delete;
        std::cout<<"re-add complete"<<std::endl;

        std::cout<<"begin search"<<std::endl;
        // Perform k-NN search and measure recall and query time
        std::vector<std::vector<size_t>> labels;
        std::vector<double> query_times;
        util::query_hnsw_single(index, queries, dim, k, labels, query_times);
        float recall = util::recall_score(ground_truth, labels, index_map, data_siz);

        std::cout<<"search complete"<<std::endl;
        double avg_query_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) / queries.size();

        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
        std::cout << "RECALL: " << recall << "\n";

        std::string avg_query_time_string = std::to_string(avg_query_time);
        std::string avg_delete_time_string = std::to_string(avg_delete_time);
        std::string avg_add_time_string = std::to_string(avg_add_time);
        std::string recall_string = std::to_string(recall);

        std::vector<std::vector <std::string>> result_data = {{dataset_name , avg_query_time_string , avg_add_time_string,recall_string}};
        util::writeCSVApp(output_csv_path,result_data);

    }

}

void output_CSV(std::string index_path,
                std::vector<std::vector<float>> queries,
                std::vector<std::vector<float>> data,
                std::string dataset_name,
                std::string output_csv_path,
                std::vector<std::vector<size_t>> ground_truth,
                int k = 100,
                int ef = 500){
    int dim=data[0].size(), num_data=data.size(), num_queries=queries.size();
    size_t data_siz = data.size();
    k = ground_truth[0].size();

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    index.setEf(ef);

    size_t last_idx = 0;
    {
        std::unordered_set<size_t> delete_indices_set;

        // set 10000 delete update point,
        size_t start_idx = last_idx;
        int num_to_delete = 1000;
        std::cout<<"delete "<<num_to_delete<<" points"<<std::endl;
        // 将后一半结点的下标插入到集合中
        for (size_t idx = start_idx; idx < start_idx+num_to_delete; ++idx) {
            delete_indices_set.insert(idx);
        }
        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());
        // Save the vectors and their labels to be deleted before deleting them
        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = delete_indices[i];
            deleted_vectors[i] = data[idx];
        }
        std::cout<<"begin delete"<<std::endl;
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        util::MarkDeleteMultiThread(index, delete_indices, index_map, 1);
        std::cout<<"delete complete"<<std::endl;
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();
        auto avg_delete_time = (double) delete_duration / num_data;

        // Re-add the deleted vectors with their original labels
        std::vector<size_t> new_indices(delete_indices.size());
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = index_map[delete_indices[i]];
            size_t new_idx = (idx < num_data) ? idx + num_data : idx - num_data;
            new_indices[i] = new_idx;
            index_map[delete_indices[i]] = new_idx;
        }

        std::cout<<"begin re-add"<<std::endl;
        auto start_time_add = std::chrono::high_resolution_clock::now();
        util::addPointsMultiThread(index, deleted_vectors, new_indices, 1);
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();
        double avg_add_time = (double) add_duration / num_to_delete;
        std::cout<<"re-add complete"<<std::endl;

        std::cout<<"begin search"<<std::endl;
        // Perform k-NN search and measure recall and query time
        std::vector<std::vector<size_t>> labels;
        std::vector<double> query_times;
        util::query_hnsw_single(index, queries, dim, k, labels, query_times);
        float recall = util::recall_score(ground_truth, labels, index_map, data_siz);
        std::cout<<"search complete"<<std::endl;
        double avg_query_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) / query_times.size();

        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
        std::cout << "RECALL: " << recall << "\n";

        std::string avg_query_time_string = std::to_string(avg_query_time);
        std::string avg_delete_time_string = std::to_string(avg_delete_time);
        std::string avg_add_time_string = std::to_string(avg_add_time);
        std::string recall_string = std::to_string(recall);

        std::vector<std::vector <std::string>> result_data = {{dataset_name , avg_query_time_string , avg_add_time_string,recall_string}};
        util::writeCSVApp(output_csv_path,result_data);

    }

}


int main(){

    std::string root_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement";

    std::vector <std::string> data_path_vec={
            root_path + "/data/sift/sift_base.fvecs",
            root_path + "/data/gist/gist_base.fvecs",
            root_path + "/data/imageNet/image.ds"
    };
    std::vector <std::string> query_data_path_vec={
            root_path + "/data/sift/sift_query.fvecs",
            root_path + "/data/gist/gist_query.fvecs",
            root_path + "/data/imageNet/image.q"
    };
    std::vector <std::string> index_path_vec={
            root_path + "/data/sift/hnsw_prime/sift_hnsw_prime_index.bin",
            root_path + "/data/gist/hnsw_prime/gist_hnsw_prime_index.bin",
            root_path + "/data/imageNet/hnsw_prime/imageNet_hnsw_prime_index.bin"
    };

    std::vector <std::string> dataset_name_vec={
            "sift_1M",
            "gist_1M",
            "imageNet"
    };

    std::vector <std::string> ground_truth_path_vec={
            root_path + "/data/sift/sift_groundtruth.ivecs",
            root_path + "/data/gist/gist_groundtruth.ivecs",
            root_path + "/data/imageNet/imageNet_groundtruth.ivecs"
    };

//    std::string data_path = root_path + "/data/sift/sift_base.fvecs";
//    std::string query_data_path = root_path + "/data/sift/sift_query.fvecs";
//    std::string index_path = root_path + "/data/sift/direct_delete/sift_base_all.bin";
//    std::string ground_truth_path = root_path + "/data/sift/sift_groundtruth.ivecs";
    std::string output_csv_path = root_path + "/output/table_1/compare_queryTime_and_deleteUpdateTime";

    // generate CSV file
    std::vector<std::vector <std::string>> header = {{"dataset_name" , "query_time" , "delete_update_time", "recall"}};
    util::writeCSVOut(output_csv_path,header);

    int siz = data_path_vec.size();
    for(int i = 0 ; i < siz ; i++){
        std::string data_path = data_path_vec[i];
        std::string query_data_path = query_data_path_vec[i];
        std::string index_path = index_path_vec[i];
        std::string dataset_name = dataset_name_vec[i];
        std::string ground_truth_path = ground_truth_path_vec[i];
        std::cout<<"-------------------begin "+ dataset_name + " generation-------------------"<<std::endl;
        int ef;
        switch (i) {
            case 0:{
                ef=30;
            }
            break;
            case 1:{
                ef=400;
            }
            break;
            default:{
                ef = 2000;
            }
        }
        output_CSV(index_path,query_data_path,data_path,dataset_name,output_csv_path,ground_truth_path,ef);
        std::cout<<"-------------------"+ dataset_name + " end-------------------"<<std::endl;
    }
//    util::writeCSV("/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/log/generate_groundtruth.log",header);
    return 0;
}