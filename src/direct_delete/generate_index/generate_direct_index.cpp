#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <exception>
#include <mutex>
#include "hnswlib/hnswlib.h"

// Function to load data from .fvecs file
std::vector<std::vector<float>> load_fvecs(const std::string& filename, int& dim, int& num) {
    std::ifstream input(filename, std::ios::binary);
    input.read(reinterpret_cast<char*>(&dim), sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num = file_size / ((dim + 1) * sizeof(float));
    input.seekg(0, std::ios::beg);

    std::vector<std::vector<float>> data(num, std::vector<float>(dim));
    for (int i = 0; i < num; ++i) {
        input.ignore(sizeof(int)); // ignore dimension
        input.read(reinterpret_cast<char*>(data[i].data()), dim * sizeof(float));
    }

    return data;
}

// Multithreaded executor
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

        // keep track of exceptions in threads
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

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

int main() {
    std::string data_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
    std::string index_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/direct_delete/direct_delete.bin";

    int dim, num_points;
    std::vector<std::vector<float>> data = load_fvecs(data_path, dim, num_points);



    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> appr_alg(&space, num_points, 16, 200);

//    num_points =500000;

//    for(int i = 0 ;i < data.size() ; i ++){
//        appr_alg.checkTotalInOutDegreeEquality();
//        appr_alg.addPoint(data[i].data(),i);
//    }

    size_t numThreads = std::thread::hardware_concurrency();
    ParallelFor(0, num_points, numThreads, [&](size_t id, size_t threadId) {
        appr_alg.addPoint(data[id].data(), id);
    });

    appr_alg.checkTotalInOutDegreeEquality();

    appr_alg.saveIndex(index_path);

    std::cout << "HNSW index created and saved to " << index_path << std::endl;

//    hnswlib::HierarchicalNSW<float> appr_alg2(&space, index_path, false, num_points, true);

//    std::cout<<"ok"<<std::endl;
    return 0;
}

//int main(){
//    std::unordered_set <int> set1;
//    set1.insert(12);
//    set1.insert(22);
//    set1.insert(33);
//    set1.insert(44);
//    set1.insert(55);
//    set1.erase(33);
//    set1.erase(4);
//    std::cout<<1<<std::endl;
//    return 0;
//}