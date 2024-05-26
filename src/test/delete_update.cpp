//
// Created by root on 5/13/24.
//
//#program omp

//#include <bits/stdc++.h>
//#include "hnswlib/hnswlib.h"
//
////using namespace hnswlib;
//
//// 直接测试每个连通块有多少个点
//
//std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
//    std::ifstream input(filename, std::ios::binary);
//    std::vector<std::vector<float>> vectors;
//    if (!input.is_open()) {
//        std::cerr << "Error opening file: " << filename << std::endl;
//        return vectors;
//    }
//
//    int dimension;
//    while (input.read((char*)&dimension, sizeof(int))) {
//        std::vector<float> vec(dimension);
//        input.read((char*)vec.data(), dimension * sizeof(float));
//        vectors.push_back(vec);
//    }
//    input.close();
//    return vectors;
//}
//
//template<class Function>
//inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
//    if (numThreads <= 0) {
//        numThreads = std::thread::hardware_concurrency();
//    }
//
//    if (numThreads == 1) {
//        for (size_t id = start; id < end; id++) {
//            fn(id, 0);
//        }
//    } else {
//        std::vector<std::thread> threads;
//        std::atomic<size_t> current(start);
//
//        std::exception_ptr lastException = nullptr;
//        std::mutex lastExceptMutex;
//
//        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
//            threads.push_back(std::thread([&, threadId] {
//                while (true) {
//                    size_t id = current.fetch_add(1);
//
//                    if (id >= end) {
//                        break;
//                    }
//
//                    try {
//                        fn(id, threadId);
//                    } catch (...) {
//                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
//                        lastException = std::current_exception();
//                        current = end;
//                        break;
//                    }
//                }
//            }));
//        }
//        for (auto &thread : threads) {
//            thread.join();
//        }
//        if (lastException) {
//            std::rethrow_exception(lastException);
//        }
//    }
//}
//
//void delete_and_update(int delete_start_index, int delete_end_index){
//
//}
//
//
//int main(){
//    int dim = 128;  // 确认向量的维度
//    int total_elements = 1000000;  // 假设索引最大元素数量为100万
//
//    // Assuming 'float' is the data type used in your HNSW index.
//    typedef float dist_t;
//    std::string indexLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base.bin"; // Update with your index file path
//    std::string dataLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
//    //    std::string indexLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/index/sift_index.bin"; // Update with your index file path
//
//    // Creating a space for the metric (e.g., L2 space for float vectors)
//    hnswlib::SpaceInterface<dist_t>* space = new hnswlib::L2Space(dim); // Replace `dimensions` with the correct dimension size
//
//
//    // Load the index
//    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, indexLocation, false,total_elements,true);
////    alg_hnsw->loadIndex("path_to_existing_index.bin", space);
//    auto vectors = read_fvecs(dataLocation);
//
//
//    int num_threads = std::thread::hardware_concurrency();
//    int num_to_replace_each_time = 100000;
//    for (int cycle = 0; cycle < 5; cycle++) {
//        int start_index_to_delete = cycle * num_to_replace_each_time;
//        int start_index_to_add = 500000 + cycle * num_to_replace_each_time;
//
//        std::cout<<"第"<<cycle<<"轮开始删除"<<std::endl;
//        ParallelFor(start_index_to_delete, start_index_to_delete + num_to_replace_each_time, num_threads, [&](size_t i, size_t threadId) {
//            alg_hnsw->markDelete(i);
//        });
//
//        std::cout<<"第"<<cycle<<"轮开始添加"<<std::endl;
//        ParallelFor(0, num_to_replace_each_time, num_threads, [&](size_t i, size_t threadId) {
//            alg_hnsw->addPoint(&(vectors[start_index_to_add + i][0]), start_index_to_delete + i, true);
//        });
//        std::cout << "第" << cycle << "轮的结果:" << std::endl;
//        // Check the connected components at each level and display the indices of components
//        std::vector<std::vector<int>> components_per_level = alg_hnsw->countConnectedComponentsPerLevel1();
//        for (int level = 0; level < components_per_level.size(); ++level) {
//            std::cout << "Level " << level << ": " << components_per_level[level].size() << " connected components." << std::endl;
//            for (int comp = 0; comp < components_per_level[level].size(); ++comp) {
//                std::cout << "  Component " << comp + 1 << ": " << components_per_level[level][comp] << " nodes" << std::endl;
//            }
//        }
//
//    }
//    return 0;
//}


#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <hnswlib/hnswlib.h>
#include <chrono>

// 加载SIFT查询数据集
std::vector<std::vector<float>> load_fvecs(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    std::vector<std::vector<float>> data;

    if (!file) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return data;
    }

    while (true) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof()) break;

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (file.eof()) break;

        data.push_back(vec);
    }

    return data;
}

// 加载HNSW索引
hnswlib::HierarchicalNSW<float>* load_hnsw_index(const std::string &index_path, int dim, int max_elements, int ef) {
    hnswlib::SpaceInterface<float> *space = new hnswlib::L2Space(dim);
    hnswlib::HierarchicalNSW<float> *index = new hnswlib::HierarchicalNSW<float>(space, index_path, false, max_elements);
    index->ef_ = ef; // 设置 ef 参数
    return index;
}

// 进行查询
std::vector<std::pair<std::vector<size_t>, std::vector<float>>> query_index(hnswlib::HierarchicalNSW<float>* index, const std::vector<std::vector<float>> &queries, int k=1) {
    std::vector<std::pair<std::vector<size_t>, std::vector<float>>> results;
    for (const auto &query : queries) {
        std::priority_queue<std::pair<float, size_t>> result = index->searchKnn(query.data(), k);

        std::vector<size_t> labels;
        std::vector<float> distances;
        while (!result.empty()) {
            labels.push_back(result.top().second);
            distances.push_back(result.top().first);
            result.pop();
        }

        results.push_back({labels, distances});
    }
    return results;
}

//std::vector<std::pair<std::vector<size_t>, std::vector<float>>> query_index(hnswlib::HierarchicalNSW<float>* index, const std::vector<std::vector<float>> &queries, int k=1) {
//    std::vector<std::pair<std::vector<size_t>, std::vector<float>>> results;
//    auto start_time = std::chrono::high_resolution_clock::now();
//
//    double total_query_time = 0.0;
//    double max_query_time = std::numeric_limits<double>::lowest();
//    double min_query_time = std::numeric_limits<double>::max();
//
//    for (const auto &query : queries) {
//        const float* query_data = query.data();  // 直接获取指针
//        auto query_start_time = std::chrono::high_resolution_clock::now();
//
//        std::priority_queue<std::pair<float, size_t>> result = index->searchKnn(query_data, k);
//
//        auto query_end_time = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> query_duration = query_end_time - query_start_time;
//
//        double query_time = query_duration.count();
//        total_query_time += query_time;
//        if (query_time > max_query_time) max_query_time = query_time;
//        if (query_time < min_query_time) min_query_time = query_time;
//
//        std::cout << "Time for one searchKnn call: " << query_time << " seconds" << std::endl;
//
//        std::vector<size_t> labels;
//        std::vector<float> distances;
//        while (!result.empty()) {
//            labels.push_back(result.top().second);
//            distances.push_back(result.top().first);
//            result.pop();
//        }
//
//        results.push_back({labels, distances});
//    }
//    auto end_time = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> total_duration = end_time - start_time;
//
//    std::cout << "Total time for all searchKnn calls: " << total_duration.count() << " seconds" << std::endl;
//    std::cout << "Total query time: " << total_query_time << " seconds" << std::endl;
//    std::cout << "Average query time: " << total_query_time / queries.size() << " seconds" << std::endl;
//    std::cout << "Max query time: " << max_query_time << " seconds" << std::endl;
//    std::cout << "Min query time: " << min_query_time << " seconds" << std::endl;
//
//    return results;
//}


std::vector<int> generate_random_numbers(size_t start_indice, size_t end_indice, const std::vector<int>& specific_queries, size_t count) {
    std::unordered_set<int> specific_queries_set(specific_queries.begin(), specific_queries.end());
    size_t max_count = (end_indice - start_indice + 1) - specific_queries.size();
    count = std::min(count, max_count);

    std::vector<int> random_numbers;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(start_indice, end_indice);

    while (random_numbers.size() < count) {
        int num = dis(gen);
        if ((specific_queries_set.find(num) == specific_queries_set.end())) {
            random_numbers.push_back(num);
        }
    }

    return random_numbers;
}

// start_indice和end_indice指示当前的索引拥有的下标范围
void test_sift_index_with_range(size_t start_indice, size_t end_indice,
                                std::vector<std::vector<float>>& sift_query,
                                const std::vector<std::vector<float>>& sift_base,
                                std::string hnsw_index_file){

    std::vector<std::vector<float>> queries_tmp = {*(sift_query.begin()+1150)};
    sift_query = queries_tmp;
    // HNSW索引参数
    int dim = sift_query[0].size();  // 数据的维度
    int max_elements = 1000000;  // HNSW索引中的最大元素数

    // 加载HNSW索引
    hnswlib::HierarchicalNSW<float>* index = load_hnsw_index(hnsw_index_file, dim, max_elements,1000000);

    // 查询HNSW索引
    int k = 1000000;  // 每个查询返回的最近邻数,设置base的数量,每个index保持五十万点的数量,按正常应该要返回所有的点
    auto results = query_index(index, sift_query, k);
    std::unordered_map <size_t,bool> excluded_global_labels_all;

    // 打印查询结果
    for (size_t i = 0; i < sift_query.size(); ++i) {
        std::cout << "Query " << i << ":" << std::endl;
        std::cout << "Labels length: " << results[i].first.size()<<",只能找到这么多的点"<< std::endl;
        auto global_labels = results[i].first;

        auto excluded_global_labels = index->getExcludedGlobalLabels(global_labels);
        for(auto &i : excluded_global_labels){
            excluded_global_labels_all[i] = true;
        }
        std::cout<<excluded_global_labels.size()<<std::endl;
        for(auto &i : excluded_global_labels){
            std::cout<<i<<" ";
            std::cout<<"它的邻居"<<std::endl;
            auto neighbours = index->getNeighborLabels(i);
            for(auto &j : neighbours){
                std::cout<<j<<" ";
            }
            std::cout<<std::endl;
        }
    }

    std::cout<<"展示结束"<<std::endl;
    std::cout<<"现在查看无法被访问点的top-1,看是否真的找不到自身"<<std::endl;
    std::vector<int> specific_queries;
    for(auto it = excluded_global_labels_all.begin();it != excluded_global_labels_all.end(); it++){
        if(it->second == true){
            specific_queries.push_back(it->first);
        }
    }
    std::vector<std::vector<float>> specific_data;
    for (int idx : specific_queries) {
        if (idx >= 0 && idx < sift_base.size()) {
            specific_data.push_back(sift_base[idx]);
        } else {
            std::cerr << "Index " << idx << " is out of bounds." << std::endl;
        }
    }
    auto specific_results = query_index(index, specific_data, 1);
    for (size_t i = 0; i < specific_results.size(); ++i) {
        std::cout << "Specific Query " << specific_queries[i] << ":" << std::endl;
        for (size_t j = 0; j < specific_results[i].first.size(); ++j) {
            std::cout << "Neighbor " << j + 1 << ": Label = " << specific_results[i].first[j] << ", Distance = " << specific_results[i].second[j] << std::endl;
        }
    }
    std::cout<<"无法被访问的点展示完毕"<<std::endl;
    std::cout<<"为了对比起见,现在展示一些可以被访问的点的top-1"<<std::endl;
    // 在这里添加逻辑,使得生成start_indice到end_indice范围的随机数,并且不包含specific_queries中的数
    size_t desired_count = 50; // 你想要生成的随机数数量
    std::vector<int> random_queries = generate_random_numbers(start_indice, end_indice, specific_queries, desired_count);
    for(auto &it : random_queries){
        std::cout<<it<<std::endl;
    }

    specific_data.clear();
    for (int idx : random_queries) {
        if (idx >= 0 && idx < sift_base.size()) {
            specific_data.push_back(sift_base[idx]);
        } else {
            std::cerr << "Index " << idx << " is out of bounds." << std::endl;
        }
    }
    specific_results = query_index(index, specific_data, 1);
    for (size_t i = 0; i < specific_results.size(); ++i) {
        std::cout << "Specific Query " << random_queries[i] << ":" << std::endl;
        for (size_t j = 0; j < specific_results[i].first.size(); ++j) {
            std::cout << "Neighbor " << j + 1 << ": Label = " << specific_results[i].first[j] << ", Distance = " << specific_results[i].second[j] << std::endl;
        }
    }
    std::cout << std::endl;
    delete index;
}


void test_sift_index_with_range_pressure(size_t start_indice, size_t end_indice,
                                std::vector<std::vector<float>>& sift_query,
                                const std::vector<std::vector<float>>& sift_base,
                                std::string hnsw_index_file){

    std::vector<std::vector<float>> queries_tmp = {*(sift_query.begin()+1150)};
    sift_query = queries_tmp;
    // HNSW索引参数
    int dim = sift_query[0].size();  // 数据的维度
    int max_elements = 1000000;  // HNSW索引中的最大元素数

    // 加载HNSW索引
    hnswlib::HierarchicalNSW<float>* index = load_hnsw_index(hnsw_index_file, dim, max_elements,2000);

    // 查询HNSW索引
    int k = 1000000;  // 每个查询返回的最近邻数,设置base的数量,每个index保持五十万点的数量,按正常应该要返回所有的点
    auto results = query_index(index, sift_query, k);
    std::unordered_map <size_t,bool> excluded_global_labels_all;

    // 打印查询结果
    for (size_t i = 0; i < sift_query.size(); ++i) {
//        std::cout << "Query " << i << ":" << std::endl;
//        std::cout << "Labels length: " << results[i].first.size()<<",只能找到这么多的点"<< std::endl;
        auto global_labels = results[i].first;
        auto excluded_global_labels = index->getExcludedGlobalLabels(global_labels);
        for(auto &i : excluded_global_labels){
            excluded_global_labels_all[i] = true;
        }
    }

    std::cout<<"展示结束"<<std::endl;
    std::cout<<"现在查看无法被访问点的top-1,看是否真的找不到自身"<<std::endl;
    std::vector<int> specific_queries;
    for(auto it = excluded_global_labels_all.begin();it != excluded_global_labels_all.end(); it++){
        if(it->second == true){
            specific_queries.push_back(it->first);
        }
    }
    std::vector<std::vector<float>> specific_data;
    for (int idx : specific_queries) {
        if (idx >= 0 && idx < sift_base.size()) {
            specific_data.push_back(sift_base[idx]);
        } else {
            std::cerr << "Index " << idx << " is out of bounds." << std::endl;
        }
    }
    auto specific_results = query_index(index, specific_data, 1);
    for (size_t i = 0; i < specific_results.size(); ++i) {
//        std::cout << "Specific Query " << specific_queries[i] << ":" << std::endl;
        for (size_t j = 0; j < specific_results[i].first.size(); ++j) {
            if(specific_results[i].second[j] == 0 && specific_results[i].first[j] == specific_queries[i]){ // 这个地方防止有别的不同标签与其是同一向量的情况
                std::cout<<"找不到的点中出现反例,搜到的标签是: "<< specific_results[i].first[j] << std::endl;
            }
        }
    }
    std::cout<<"无法被访问的点展示完毕"<<std::endl;
    std::cout<<"为了对比起见,现在展示一些可以被访问的点的top-1"<<std::endl;
    // 在这里添加逻辑,使得生成start_indice到end_indice范围的随机数,并且不包含specific_queries中的数
    size_t desired_count = 10000; // 你想要生成的随机数数量
    std::vector<int> random_queries = generate_random_numbers(start_indice, end_indice, specific_queries, desired_count);
    std::cout<<"随机数生成完毕"<<std::endl;
//    for(auto &it : random_queries){
//        std::cout<<it<<std::endl;
//    }

    specific_data.clear();
    for (int idx : random_queries) {
        if (idx >= 0 && idx < sift_base.size()) {
            specific_data.push_back(sift_base[idx]);
        } else {
            std::cerr << "Index " << idx << " is out of bounds." << std::endl;
        }
    }
    specific_results = query_index(index, specific_data, 1);
    for (size_t i = 0; i < specific_results.size(); ++i) {
//        std::cout << "Specific Query " << random_queries[i] << ":" << std::endl;
        for (size_t j = 0; j < specific_results[i].first.size(); ++j) {
            if(specific_results[i].second[j] != 0){ // 能找到的点理应在索引中找到距离为0的点
                std::cout<<"理应找到的点中出现反例(当然也有可能是ef设得太低了),出现反例的标签是: "<< random_queries[i] << std::endl;
                std::cout<<"找到的标签是: "<< specific_results[i].first[j] << std::endl;
            }
        }
    }
    std::cout << std::endl;
    delete index;
}

void test_opponent(const std::vector<std::vector<float>>& sift_base,
                   std::string hnsw_index_file){
    // HNSW索引参数
    int dim = sift_base[0].size();  // 数据的维度
    int max_elements = 1000000;  // HNSW索引中的最大元素数

    // 加载HNSW索引
    hnswlib::HierarchicalNSW<float>* index = load_hnsw_index(hnsw_index_file, dim, max_elements,1000000);

    std::vector<int> specific_queries = {615059,897480};
    std::vector<std::vector<float>> specific_data;
    std::cout<<"base_set"<<sift_base.size()<<std::endl;
    for (int idx : specific_queries) {
        if (idx >= 0 && idx < sift_base.size()) {
            specific_data.push_back(sift_base[idx]);
        } else {
            std::cerr << "Index " << idx << " is out of bounds." << std::endl;
        }
    }

    auto specific_results = query_index(index, specific_data, 1);
    for (size_t i = 0; i < specific_results.size(); ++i) {
        std::cout << "Specific Query " << specific_queries[i] << ":" << std::endl;
        for (size_t j = 0; j < specific_results[i].first.size(); ++j) {
            std::cout << "Neighbor " << j + 1 << ": Label = " << specific_results[i].first[j] << ", Distance = " << specific_results[i].second[j] << std::endl;
        }
    }


    delete index;
}


int main() {
    // 文件路径
    std::string sift_query_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs";
    std::string hnsw_index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half/initial_index";
    std::string hnsw_index_file1 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half/index_iteration_1";
    std::string hnsw_index_file2 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half/index_iteration_2";
    std::string hnsw_index_file3 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half/index_iteration_3";
    std::string hnsw_index_file4 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half/index_iteration_4";
    std::string hnsw_index_file5 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half/index_iteration_5";

    std::string sift_base_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
//    std::string hnsw_index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_base.bin";
    // 加载查询数据
    std::vector<std::vector<float>> queries = load_fvecs(sift_query_file);
    if (queries.empty()) {
        std::cerr << "Failed to load queries." << std::endl;
        return 1;
    }
    // 加载基数据
    std::vector<std::vector<float>> base_data = load_fvecs(sift_base_file);
    if (base_data.empty()) {
        std::cerr << "Failed to load base data." << std::endl;
        return 1;
    }
    test_sift_index_with_range(0,499999,queries,base_data,hnsw_index_file);
    std::cout<<"-----------------------------------没有任何删除添加的index展示完毕-----------------------------------"<<std::endl;
    test_sift_index_with_range(100000,599999,queries,base_data,hnsw_index_file1);
    std::cout<<"-----------------------------------hnsw_index_file1展示完毕-----------------------------------"<<std::endl;
    test_sift_index_with_range(200000,699999,queries,base_data,hnsw_index_file2);
    std::cout<<"-----------------------------------hnsw_index_file2展示完毕-----------------------------------"<<std::endl;
    test_sift_index_with_range(300000,799999,queries,base_data,hnsw_index_file3);
    std::cout<<"-----------------------------------hnsw_index_file3展示完毕-----------------------------------"<<std::endl;
    test_sift_index_with_range(400000,899999,queries,base_data,hnsw_index_file4);
    std::cout<<"-----------------------------------hnsw_index_file4展示完毕-----------------------------------"<<std::endl;
    test_sift_index_with_range(500000,999999,queries,base_data,hnsw_index_file5);
    std::cout<<"-----------------------------------hnsw_index_file5展示完毕-----------------------------------"<<std::endl;

//    test_sift_index_with_range_pressure(0,499999,queries,base_data,hnsw_index_file);
//    std::cout<<"-----------------------------------没有任何删除添加的index展示完毕-----------------------------------"<<std::endl;
//    test_sift_index_with_range_pressure(100000,599999,queries,base_data,hnsw_index_file1);
//    std::cout<<"-----------------------------------hnsw_index_file1展示完毕-----------------------------------"<<std::endl;
//    test_sift_index_with_range_pressure(200000,699999,queries,base_data,hnsw_index_file2);
//    std::cout<<"-----------------------------------hnsw_index_file2展示完毕-----------------------------------"<<std::endl;
//    test_sift_index_with_range_pressure(300000,799999,queries,base_data,hnsw_index_file3);
//    std::cout<<"-----------------------------------hnsw_index_file3展示完毕-----------------------------------"<<std::endl;
//    test_sift_index_with_range_pressure(400000,899999,queries,base_data,hnsw_index_file4);
//    std::cout<<"-----------------------------------hnsw_index_file4展示完毕-----------------------------------"<<std::endl;
//    test_sift_index_with_range_pressure(500000,999999,queries,base_data,hnsw_index_file5);
//    std::cout<<"-----------------------------------hnsw_index_file5展示完毕-----------------------------------"<<std::endl;

//    test_opponent(base_data,hnsw_index_file5);


//    std::vector<std::vector<float>> queries_tmp = {*(queries.begin()+1150)};
//    queries = queries_tmp;
//    // HNSW索引参数
//    int dim = queries[0].size();  // 数据的维度
//    int max_elements = 1000000;  // HNSW索引中的最大元素数
//
//    // 加载HNSW索引
//    hnswlib::HierarchicalNSW<float>* index = load_hnsw_index(hnsw_index_file, dim, max_elements);
//
//    // 查询HNSW索引
//    int k = 1000000;  // 每个查询返回的最近邻数
//    auto results = query_index(index, queries, k);
//    std::unordered_map <size_t,bool> excluded_global_labels_all;
//
//    // 打印查询结果
//    for (size_t i = 0; i < queries.size(); ++i) {
//        std::cout << "Query " << i << ":" << std::endl;
//        std::cout << "Labels length: " << results[i].first.size()<<",只能找到这么多的点"<< std::endl;
//        auto global_labels = results[i].first;
//
//        auto excluded_global_labels = index->getExcludedGlobalLabels(global_labels);
//        for(auto &i : excluded_global_labels){
//            excluded_global_labels_all[i] = true;
//        }
//        std::cout<<excluded_global_labels.size()<<std::endl;
//        for(auto &i : excluded_global_labels){
//            std::cout<<i<<" ";
//            std::cout<<"它的邻居"<<std::endl;
//            auto neighbours = index->getNeighborLabels(i);
//            for(auto &j : neighbours){
//                std::cout<<j<<" ";
//            }
//            std::cout<<std::endl;
//        }
//    }
//    std::cout<<"展示结束"<<std::endl;
//
////    // 这个地方说明建边不是双向的
////    std::cout<<448331<<" ";
////    std::cout<<"它的邻居"<<std::endl;
////    auto neighbours = index->getNeighborLabels(448331);
////    for(auto &j : neighbours){
////        std::cout<<j<<" ";
////    }
////    std::cout<<std::endl;
////
////    std::cout<<45655<<" ";
////    std::cout<<"它的邻居"<<std::endl;
////    neighbours = index->getNeighborLabels(45655);
////    for(auto &j : neighbours){
////        std::cout<<j<<" ";
////    }
////    std::cout<<std::endl;
////
////    std::cout<<134573<<" ";
////    std::cout<<"它的邻居"<<std::endl;
////    neighbours = index->getNeighborLabels(134573);
////    for(auto &j : neighbours){
////        std::cout<<j<<" ";
////    }
////    std::cout<<std::endl;
//
//    std::cout<<"现在查看无法被访问点"<<std::endl;
//    // 查询特定数据点
//    std::vector<int> specific_queries = {126659,221045, 448331,218345};
//    std::vector<std::vector<float>> specific_data;
//    std::cout<<"base_set"<<base_data.size()<<std::endl;
//    for (int idx : specific_queries) {
//        if (idx >= 0 && idx < base_data.size()) {
//            specific_data.push_back(base_data[idx]);
//        } else {
//            std::cerr << "Index " << idx << " is out of bounds." << std::endl;
//        }
//    }
//
//    auto specific_results = query_index(index, specific_data, 1);
//    for (size_t i = 0; i < specific_results.size(); ++i) {
//        std::cout << "Specific Query " << specific_queries[i] << ":" << std::endl;
//        for (size_t j = 0; j < specific_results[i].first.size(); ++j) {
//            std::cout << "Neighbor " << j + 1 << ": Label = " << specific_results[i].first[j] << ", Distance = " << specific_results[i].second[j] << std::endl;
//        }
//    }
//
//    delete index;
    return 0;
}


// 试一下看最终无法找到的点与最底层的



//// 这是128维度的删除更新测试
//
//#include <bits/stdc++.h>
//#include "hnswlib/hnswlib.h"
//#include <thread>
//#include <atomic>
//
//std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
//    std::ifstream input(filename, std::ios::binary);
//    std::vector<std::vector<float>> vectors;
//    if (!input.is_open()) {
//        std::cerr << "Error opening file: " << filename << std::endl;
//        return vectors;
//    }
//
//    int dimension;
//    while (input.read((char*)&dimension, sizeof(int))) {
//        std::vector<float> vec(dimension);
//        input.read((char*)vec.data(), dimension * sizeof(float));
//        vectors.push_back(vec);
//    }
//    input.close();
//    return vectors;
//}
//
//template<class Function>
//inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
//    if (numThreads <= 0) {
//        numThreads = std::thread::hardware_concurrency();
//    }
//
//    if (numThreads == 1) {
//        for (size_t id = start; id < end; id++) {
//            fn(id, 0);
//        }
//    } else {
//        std::vector<std::thread> threads;
//        std::atomic<size_t> current(start);
//
//        std::exception_ptr lastException = nullptr;
//        std::mutex lastExceptMutex;
//
//        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
//            threads.push_back(std::thread([&, threadId] {
//                while (true) {
//                    size_t id = current.fetch_add(1);
//
//                    if (id >= end) {
//                        break;
//                    }
//
//                    try {
//                        fn(id, threadId);
//                    } catch (...) {
//                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
//                        lastException = std::current_exception();
//                        current = end;
//                        break;
//                    }
//                }
//            }));
//        }
//        for (auto &thread : threads) {
//            thread.join();
//        }
//        if (lastException) {
//            std::rethrow_exception(lastException);
//        }
//    }
//}
//
//
//int main() {
//    int dim = 128;  // Dimension of the elements
//    int total_elements = 1000000;  // Maximum number of elements
//    int num_threads = std::thread::hardware_concurrency();  // Number of threads
//
//    typedef float dist_t;
////    std::string indexLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift_1w/index/hnsw_index_l2.bin"; // Update with your index file path
//    std::string dataLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
////    std::string indexLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/index/sift_index.bin"; // Update with your index file path
//    std::string indexLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_delete_over.bin"; // Update with your index file path
//
//
//    hnswlib::SpaceInterface<dist_t>* space = new hnswlib::L2Space(dim); // Replace `dimensions` with the correct dimension size
//    // Load the index
//    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, indexLocation, false,total_elements,true);
//    auto vectors = read_fvecs(dataLocation);
//
////    std::vector<std::vector<int>> components_per_level = alg_hnsw->countConnectedComponentsPerLevel();
////    for (int level = 0; level < components_per_level.size(); ++level) {
////        std::cout << "Level " << level << ": " << components_per_level[level].size() << " connected components." << std::endl;
////        for (int comp = 0; comp < components_per_level[level].size(); ++comp) {
////            std::cout << "  Component " << comp + 1 << ": " << components_per_level[level][comp] << " nodes" << std::endl;
////        }
////    }
//
//    // tarjan一百万个点会爆内存貌似,很奇怪.
//    std::vector<std::vector<std::vector<hnswlib::labeltype>>> components_per_level = alg_hnsw->countConnectedComponentsPerLevel();
//
//    // 输出每一层的连通分量信息
//    for (int level = 0; level <= alg_hnsw->maxlevel_; ++level) {
//        std::cout << "Level " << level << ":" << std::endl;
//        for (const auto& component : components_per_level[level]) {
//            std::cout << "Component size: " << component.size() << ", Labels: ";
//            for (hnswlib::labeltype label : component) {
//                std::cout << label << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//    std::cout<<1<<std::endl;
//
//
//    int num_to_replace_each_time = 100000;
//    for (int cycle = 0; cycle < 5; cycle++) {
//        int start_index_to_delete = cycle * num_to_replace_each_time;
//        int start_index_to_add = 500000 + cycle * num_to_replace_each_time;
//
//        std::cout<<"第"<<cycle<<"轮开始删除"<<std::endl;
//        ParallelFor(start_index_to_delete, start_index_to_delete + num_to_replace_each_time, num_threads, [&](size_t i, size_t threadId) {
//            alg_hnsw->markDelete(i);
//        });
//
//        std::cout<<"第"<<cycle<<"轮开始添加"<<std::endl;
//        ParallelFor(0, num_to_replace_each_time, num_threads, [&](size_t i, size_t threadId) {
//            alg_hnsw->addPoint(&(vectors[start_index_to_add + i][0]), start_index_to_delete + i, true);
//        });
//        std::cout << "第" << cycle << "轮的结果:" << std::endl;
//        // Check the connected components at each level and display the indices of components
//        std::vector<std::vector<std::vector<hnswlib::labeltype>>> components_per_level = alg_hnsw->countConnectedComponentsPerLevel();
//        for (int level = 0; level < components_per_level.size(); ++level) {
//            std::cout << "Level " << level << ": " << components_per_level[level].size() << " connected components." << std::endl;
//            for (int comp = 0; comp < components_per_level[level].size(); ++comp) {
//                std::cout << "  Component " << comp + 1 << ": " << components_per_level[level][comp].size() << " nodes, Indices: ";
//                for (int idx = 0; idx < components_per_level[level][comp].size(); ++idx) {
//                    std::cout << components_per_level[level][comp][idx];
//                    if (idx < components_per_level[level][comp].size() - 1) std::cout << ", ";
//                }
//                std::cout << std::endl;
//            }
//        }
//    }
////    std::string save_index_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_index/sift_delete_over.bin";
////    alg_hnsw->saveIndex(save_index_path);
//    delete alg_hnsw;
//    delete space;
//    return 0;
//}



// 这是2D的索引测试

//#include <bits/stdc++.h>
//#include "hnswlib/hnswlib.h"
//#include <thread>
//#include <atomic>
//
//std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
//    std::ifstream input(filename, std::ios::binary);
//    std::vector<std::vector<float>> vectors;
//    if (!input.is_open()) {
//        std::cerr << "Error opening file: " << filename << std::endl;
//        return vectors;
//    }
//
//    int dimension;
//    while (input.read((char*)&dimension, sizeof(int))) {
//        std::vector<float> vec(dimension);
//        input.read((char*)vec.data(), dimension * sizeof(float));
//        vectors.push_back(vec);
//    }
//    input.close();
//    return vectors;
//}
//
//template<class Function>
//inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
//    if (numThreads <= 0) {
//        numThreads = std::thread::hardware_concurrency();
//    }
//
//    if (numThreads == 1) {
//        for (size_t id = start; id < end; id++) {
//            fn(id, 0);
//        }
//    } else {
//        std::vector<std::thread> threads;
//        std::atomic<size_t> current(start);
//
//        std::exception_ptr lastException = nullptr;
//        std::mutex lastExceptMutex;
//
//        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
//            threads.push_back(std::thread([&, threadId] {
//                while (true) {
//                    size_t id = current.fetch_add(1);
//
//                    if (id >= end) {
//                        break;
//                    }
//
//                    try {
//                        fn(id, threadId);
//                    } catch (...) {
//                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
//                        lastException = std::current_exception();
//                        current = end;
//                        break;
//                    }
//                }
//            }));
//        }
//        for (auto &thread : threads) {
//            thread.join();
//        }
//        if (lastException) {
//            std::rethrow_exception(lastException);
//        }
//    }
//}
//
//int main() {
//    int dim = 2;  // Dimension of the elements
//    int total_elements = 1000000;  // Maximum number of elements
//    int num_threads = std::thread::hardware_concurrency();  // Number of threads
//
//    typedef float dist_t;
//    std::string indexLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_index/delete_update.bin"; // Update with your index file path
//    std::string baseDataLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_update_base.fvecs";
//    std::string addDataLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_update_add.fvecs";
//    //    std::string indexLocation = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/index/sift_index.bin"; // Update with your index file path
//
//    hnswlib::SpaceInterface<dist_t>* space = new hnswlib::L2Space(dim); // Replace `dimensions` with the correct dimension size
//    // Load the index
//    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, indexLocation, false,total_elements,true);
//    auto base_vector = read_fvecs(baseDataLocation);
//    auto add_vectors = read_fvecs(addDataLocation);
//
//    int delete_interval = 1000;  // Delete after every 50,000 vectors
//    int delete_batch_size = 500;  // Delete 10,000 vectors at a time
//
//    for (int start_index = 0; start_index < base_vector.size(); start_index += delete_interval + delete_batch_size) {
//        int end_index = start_index + delete_batch_size;
//
//        // Mark to delete
//        std::cout << "Deleting from index " << start_index << " to " << end_index << std::endl;
//        ParallelFor(start_index, end_index, num_threads, [&](size_t i, size_t threadId) {
//            alg_hnsw->markDelete(i);
//        });
//
//        // Add new vectors
//        std::cout << "Adding new vectors starting from index " << start_index << std::endl;
//        ParallelFor(0, delete_batch_size, num_threads, [&](size_t i, size_t threadId) {
//            if (start_index + i < add_vectors.size()) {
//                alg_hnsw->addPoint(&(add_vectors[start_index + i][0]), start_index + i, true);
//            }
//        });
//        std::vector<std::vector<int>> components_per_level = alg_hnsw->countConnectedComponentsPerLevel();
//        for (int level = 0; level < components_per_level.size(); ++level) {
//            std::cout << "Level " << level << ": " << components_per_level[level].size() << " connected components." << std::endl;
//            for (int comp = 0; comp < components_per_level[level].size(); ++comp) {
//                std::cout << "  Component " << comp + 1 << ": " << components_per_level[level][comp] << " nodes" << std::endl;
//            }
//        }
//
//    }
//
//
//
//
//    delete alg_hnsw;
//    delete space;
//    return 0;
//}
