#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <exception>

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

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragma omp parallel for or any other C++ threading
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
        // https://stackoverflow.com/a/32428427/1713196
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
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
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
    const int dim = 128; // SIFT通常维度为128
    const std::string indexFile1 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/new_level_selection_index/sift_index_new_level_selection_prime_100000.bin";
//    const std::string indexFile2 = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/new_level_selection_index/sift_index_new_level_selection_100000.bin";
    const std::string queryFile = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_query.fvecs";

    hnswlib::L2Space space(dim);

    hnswlib::HierarchicalNSW<float> index1(&space, indexFile1, false);
//    hnswlib::HierarchicalNSW<float> index2(&space, indexFile2, false);

    std::vector<std::vector<float>> queries;
    if (!ReadFvecsFileIntoArray<float>(queryFile, queries, dim)) {
        std::cerr << "Failed to load query file." << std::endl;
        return -1;
    }

    auto searchIndex = [&](hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, size_t numThreads) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<hnswlib::labeltype> neighbors(queries.size());
        ParallelFor(0, queries.size(), numThreads, [&](size_t row, size_t threadId) {
            auto result = index.searchKnn(queries[row].data(), 1);
            neighbors[row] = result.top().second;
        });

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    };

    std::cout << "Searching new..." << std::endl;
    double total_time = 0.0;
    const int iterations = 10000;
    const int numThreads = std::thread::hardware_concurrency();

    for (int i = 0; i < iterations; ++i) {
//        total_time += searchIndex(index2, queries, numThreads);
        total_time += searchIndex(index1, queries,numThreads);
    }

    double average_time = total_time / iterations;
    std::cout << "Average query time over " << iterations << " runs: " << average_time << " ms" << std::endl;

    return 0;
}
