//
// Created by root on 3/25/24.
//

#include <filesystem>

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"
#include "file.h"
#include "evaluate/evaluate.h"

void generateIndexFromBvec(int dim,
                           int max_elements,
                           std::string data_file_path,
                           std::string index_file,
                           std::string log_file_path,
                           int M=16,
                           int ef_construction=200){
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    float* data = new float[dim * max_elements];
    int totalVectors = 0;
    if(ReadOpt::ReadFvecsFileIntoArray<float>(data_file_path,data,totalVectors,dim,max_elements)){
        std::cout<<"开始生成索引"<<std::endl;
        for (int i = 0; i < max_elements; i++) {
            alg_hnsw->addPoint(data + i * dim, i);
        }
        alg_hnsw->saveIndex(index_file);
        std::cout<<"索引生成完成"<<std::endl;
        delete[] data;
        delete alg_hnsw;
        WriteOpt::WriteIndexLogFile(log_file_path,data_file_path,index_file,dim,totalVectors);
    }
}

void generateIndexFromRandom(int dim,
                   int max_elements,
                   std::string data_file_path,
                   int M,
                   int ef_construction,
                   std::string index_file){
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    float* data = new float[dim * max_elements];
    std::ifstream file(data_file_path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data), dim * max_elements * sizeof(float));
    std::cout<<"开始生成索引"<<std::endl;
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }
    alg_hnsw->saveIndex(index_file);
    std::cout<<"索引生成完成"<<std::endl;
    delete[] data;
    delete alg_hnsw;
}

//生成索引
int main(int argc, char* argv[]){
    // 检查参数数量
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <dimension> <num_points> <data_file_path> <index_file_path>" << std::endl;
        return -1;
    }

    int dim = std::stoi(argv[1]);
    int max_elements = std::stoi(argv[2]);
    std::string data_file_path = argv[3];
    std::string index_file_path = argv[4];

    // 检查数据文件路径是否存在
    if (!std::filesystem::exists(data_file_path)) {
        std::cerr << "File does not exist: " << data_file_path << std::endl;
        return 1;
    }

    // 检查数据文件路径是否确实是一个文件而不是目录
    if (!std::filesystem::is_regular_file(data_file_path)) {
        std::cerr << "Path is not a file: " << data_file_path << std::endl;
        return 1;
    }

//    // 检查索引文件路径是否存在
//    if (!std::filesystem::exists(index_file_path)) {
//        std::cerr << "File does not exist: " << index_file_path << std::endl;
//        return 1;
//    }
//
//    // 检查索引文件路径是否确实是一个文件而不是目录
//    if (!std::filesystem::is_regular_file(index_file_path)) {
//        std::cerr << "Path is not a file: " << index_file_path << std::endl;
//        return 1;
//    }

    // 获取index_file_path的父目录
    std::filesystem::path index_path = index_file_path;
    std::filesystem::path log_file_directory = index_path.parent_path();
    std::string log_file_path = log_file_directory.string() + "/index_log.csv";
//    generateIndexFromRandom(dim,max_elements,data_file_path,16,200,index_file_path);
    generateIndexFromBvec(dim,max_elements,data_file_path,index_file_path,log_file_path);
}