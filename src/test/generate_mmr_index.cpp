//
// Created by root on 5/13/24.
//

#include <filesystem>

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <exception>

template<typename T>
bool ReadFvecsFileIntoArray(const std::string& filePath, void* p, int& totalVectors, const int& dim, const int &maxElements) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    int vectorDim = -1;
    // 尝试读取第一个向量的维度
    file.read(reinterpret_cast<char*>(&vectorDim), sizeof(int));
    if (vectorDim <= 0 || vectorDim != dim) {
        std::cerr << "Invalid vector dimension or dimension exceeds maximum allowed size." << std::endl;
        return false;
    }

    T* array = reinterpret_cast<T*>(p); // 将void*转换为T*，以便按T类型读取和存储数据

    // 重新定位到文件开头
    file.seekg(0, std::ios::beg);

    int vectorCount = 0;
    while (true) {
        if (!file.read(reinterpret_cast<char*>(&vectorDim), sizeof(int))) {
            // 文件结束或读取错误
            break;
        }
        if (vectorDim != dim) {
            std::cerr << "dim is not consistent" << std::endl;
            return false;
        }
        if (vectorCount >= maxElements) {
            std::cout << "Array size exceeded, but you can use the data that has been loaded" << std::endl;
            totalVectors = vectorCount;
            file.close();
            return true;
        }
        if (!file.read(reinterpret_cast<char*>(array + vectorCount * dim), sizeof(T) * dim)) {
            // 数据读取不完整
            std::cerr << "Incomplete data read." << std::endl;
            return false;
        }
        vectorCount++;
    }

    totalVectors = vectorCount;
    file.close();
    return true;
}

template<class T>
void WriteCSVFile(const std::vector<std::vector<T> >& data,const std::string & filePath) {
//    std::vector <std::vector<std::string>> coordinate={{"X","Y","Z"}}; //3D
    std::vector <std::vector<std::string>> coordinate={{"X","Y"}}; //2D
    std::ofstream outfile(filePath);
    if (outfile.is_open()) { // 检查文件是否成功打开
        for(const auto& row : coordinate){
            for(auto record: row){
                outfile << record<<",";
            }
            outfile<<"\n";
        }
        for(const auto& row : data){
            for(auto record: row){
                outfile << record<<",";
            }
            outfile<<"\n";
        }
        outfile.close(); // 关闭文件
        std::cout << "data successfully write in:"<<filePath<<".\n";
    } else {
        std::cout << "Unable to open file.\n";
    }
}


void WriteIndexLogFile(const std::string log_info_data_path,const std::string &data_file_path,const std::string &index_file_path,const int &dim, const int &vector_num){
    std::vector <std::vector<std::string>> info;
    std::vector<std::string> tmp = {"index_file_path",index_file_path};
    info.push_back(tmp);
    tmp = {"data_file_path",data_file_path};
    info.push_back(tmp);
    tmp = {"dim",std::to_string(dim)};
    info.push_back(tmp);
    tmp = {"vector_num",std::to_string(vector_num)};
    info.push_back(tmp);
    WriteCSVFile<std::string>(info,log_info_data_path);
}



// Helper function for multithreading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);
                    if (id >= end) break;
                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }

        for (auto &thread : threads) {
            thread.join();
        }

        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

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
    if(ReadFvecsFileIntoArray<float>(data_file_path,data,totalVectors,dim,max_elements)){
        std::cout<<"开始生成索引"<<std::endl;
        int num_threads = std::thread::hardware_concurrency();
        ParallelFor(0, max_elements, num_threads, [&](size_t i, size_t threadId) {
            alg_hnsw->addPoint(data + i * dim, i);
        });
        alg_hnsw->saveIndex(index_file);
        std::cout<<"索引生成完成"<<std::endl;
        delete[] data;
        delete alg_hnsw;
        WriteIndexLogFile(log_file_path,data_file_path,index_file,dim,totalVectors);
    }
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