//
// Created by root on 5/13/24.
//

#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

template<typename T>
bool ReadFvecsFileIntoArray(const std::string& filePath, std::vector<std::vector<T>>& data, const int& dim) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    int vectorDim;
    while (file.read(reinterpret_cast<char*>(&vectorDim), sizeof(int))) {
        if (vectorDim != dim) {
            std::cerr << "Vector dimension mismatch." << std::endl;
            return false;
        }
        std::vector<T> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), sizeof(T) * dim);
        data.push_back(vec);
    }
    file.close();
    return true;
}


int main() {
    const int dim = 128; // SIFT通常维度为128
//    const std::string indexFile1 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/index/sift_index_mmr.bin";
    const std::string indexFile2 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/delete_back_half/initial_index";
    const std::string queryFile = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs";

    hnswlib::L2Space space(dim);

    hnswlib::HierarchicalNSW<float> index2(&space, indexFile2, false);

    std::vector<std::vector<float>> queries;
    if (!ReadFvecsFileIntoArray<float>(queryFile, queries, dim)) {
        std::cerr << "Failed to load query file." << std::endl;
        return -1;
    }

    auto searchIndex = [&](hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries) {
        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& query : queries) {
            auto result = index.searchKnn(query.data(), 1); // 查询最近的一个点
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Total query time: " << elapsed.count() << " ms" << std::endl;
    };

    std::cout << "Searching normal..." << std::endl;
    searchIndex(index2, queries);

    return 0;
}


//template<typename T>
//bool ReadFvecsFileIntoArray(const std::string& filePath, std::vector<std::vector<T>>& data, const int& dim) {
//    std::ifstream file(filePath, std::ios::binary);
//    if (!file.is_open()) {
//        std::cerr << "Failed to open file: " << filePath << std::endl;
//        return false;
//    }
//    int vectorDim;
//    while (file.read(reinterpret_cast<char*>(&vectorDim), sizeof(int))) {
//        if (vectorDim != dim) {
//            std::cerr << "Vector dimension mismatch." << std::endl;
//            return false;
//        }
//        std::vector<T> vec(dim);
//        file.read(reinterpret_cast<char*>(vec.data()), sizeof(T) * dim);
//        data.push_back(vec);
//    }
//    file.close();
//    return true;
//}
//
//int main() {
//    const int dim = 128; // SIFT通常维度为128
//    const std::string indexFile1 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/index/sift_index_mmr.bin";
//    const std::string indexFile2 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/index/sift_index.bin";
//    const std::string queryFile = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs";
//
//    hnswlib::L2Space space(dim);
//
//    hnswlib::HierarchicalNSW<float> index1(&space, indexFile1, false);
//    hnswlib::HierarchicalNSW<float> index2(&space, indexFile2, false);
//
//    std::vector<std::vector<float>> queries;
//    if (!ReadFvecsFileIntoArray<float>(queryFile, queries, dim)) {
//        std::cerr << "Failed to load query file." << std::endl;
//        return -1;
//    }
//
//    auto searchIndex = [&](hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries) {
//        auto start = std::chrono::high_resolution_clock::now();
//        for (const auto& query : queries) {
//            auto result = index.searchKnn(query.data(), 10); // 查询最近的一个点
//        }
//        auto end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double, std::milli> elapsed = end - start;
//        std::cout << "Total query time: " << elapsed.count() << " ms" << std::endl;
//    };
//
//    std::cout << "Searching mmr..." << std::endl;
//    searchIndex(index1, queries);
//
//    std::cout << "Searching normal..." << std::endl;
//    searchIndex(index2, queries);
//
//    return 0;
//}
