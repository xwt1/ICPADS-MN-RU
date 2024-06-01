//
// Created by root on 5/27/24.
//

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

//    std::cout<<1<<std::endl;

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


int main() {

    // 文件路径
    std::string sift_query_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs";

    std::string hnsw_index_root_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/issue_statement_index/delete_update";

    std::string hnsw_index_file = hnsw_index_root_path + "/initial_index";
    std::string hnsw_index_file1 = hnsw_index_root_path + "/index_iteration_1";
    std::string hnsw_index_file2 = hnsw_index_root_path + "/index_iteration_2";
    std::string hnsw_index_file3 = hnsw_index_root_path + "/index_iteration_3";
    std::string hnsw_index_file4 = hnsw_index_root_path + "/index_iteration_4";
    std::string hnsw_index_file5 = hnsw_index_root_path + "/index_iteration_5";

    std::string hnsw_index_file100 = hnsw_index_root_path + "/index_iteration_100";

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

//    test_sift_index_with_range(500000,999999,queries,base_data,hnsw_index_file100);
//    std::cout<<"-----------------------------------hnsw_index_file100展示完毕-----------------------------------"<<std::endl;

    return 0;
}