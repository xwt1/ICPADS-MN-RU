//
// Created by root on 6/6/24.
//

#ifndef GRAPH_SEARCH_UTIL_H
#define GRAPH_SEARCH_UTIL_H
#include "iostream"
#include "vector"
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "hnswlib/hnswlib.h"
#include "thread_pool.h"

class util{
public:
    static std::vector<std::vector<float>> load_fvecs(const std::string& filename, int& dim, int& num);
//    std::string readJsonFile(const std::string& filename, const std::string& key);
    static std::vector<std::vector<size_t>> load_ivecs_indices(const std::string& filename);

//    static void query_hnsw(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, int num_threads, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times);
    static void query_hnsw(hnswlib::HierarchicalNSW<float>& alg_hnsw, const std::vector<std::vector<float>>& queries, int k, int num_threads, std::vector<std::vector<size_t>>& results);

    static void markDeleteMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<size_t>& delete_indices, const std::unordered_map<size_t, size_t>& index_map, int num_threads);

    static void addPointsMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& points, const std::vector<size_t>& labels, int num_threads);

    static void query_hnsw_single(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times);

    static std::vector<std::pair<std::vector<size_t>, std::vector<float>>> query_index(hnswlib::HierarchicalNSW<float>* index, const std::vector<std::vector<float>> &queries, int k=1);

    static void writeCSVOut(const std::string& filename, const std::vector<std::vector<std::string>>& data);

    static void writeCSVApp(const std::string& filename, const std::vector<std::vector<std::string>>& data);

    static float recall_score(const std::vector<std::vector<size_t>>& ground_truth, const std::vector<std::vector<size_t>>& predictions, const std::unordered_map<size_t, size_t>& index_map, size_t data_size);

    static void knn_thread(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& queries, int dim, int k, size_t start_idx, size_t end_idx, std::vector<std::vector<size_t>>& indices);

    static std::vector<std::vector<size_t>> brute_force_knn(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& queries, int dim, int k);

    static void save_knn_to_ivecs(const std::string& filename, const std::vector<std::vector<size_t>>& knn_results);

    static void create_directories(const std::vector<std::string>& paths);

    static std::vector<std::vector<size_t>> generate_unique_random_numbers(int limit, int size, int num) ;

    static void save_to_fvecs(const std::string& filename, const std::vector<std::vector<int>>& data);

    static void save_to_ivecs(const std::string& filename, const std::vector<std::vector<size_t>>& data);

    static std::vector<std::vector<size_t>> load_ivecs(const std::string &filename, int &dim, int &num);

    /*
     * stamp : ef增长间隔
     */
    static std::vector<std::vector<float>> countRecallWithDiffPara(hnswlib::HierarchicalNSW<float>& index,
                                                       const std::vector<std::vector<float>> &queries,
                                                       std::vector<std::vector<size_t>> ground_truth,
                                                       std::unordered_map<size_t, size_t> index_map,
                                                       int k,
                                                       int start_ef,
                                                       int end_ef,
                                                       int stamp,
                                                       int num_threads);


    template<class Function>
    inline static void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        static ThreadPool pool(numThreads > 0 ? numThreads : std::thread::hardware_concurrency());

        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        } else {
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                pool.enqueue([&, threadId] {
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
                });
            }

            pool.waitForCompletion();

            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }
    }

//    // Multithreaded executor
//    template<class Function>
//    inline static void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
//        if (numThreads <= 0) {
//            numThreads = std::thread::hardware_concurrency();
//        }
//
//        if (numThreads == 1) {
//            for (size_t id = start; id < end; id++) {
//                fn(id, 0);
//            }
//        } else {
//            std::vector<std::thread> threads;
//            std::atomic<size_t> current(start);
//
//            // keep track of exceptions in threads
//            std::exception_ptr lastException = nullptr;
//            std::mutex lastExceptMutex;
//
//            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
//                threads.push_back(std::thread([&, threadId] {
//                    while (true) {
//                        size_t id = current.fetch_add(1);
//
//                        if (id >= end) {
//                            break;
//                        }
//
//                        try {
//                            fn(id, threadId);
//                        } catch (...) {
//                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
//                            lastException = std::current_exception();
//                            current = end;
//                            break;
//                        }
//                    }
//                }));
//            }
//            for (auto &thread : threads) {
//                thread.join();
//            }
//            if (lastException) {
//                std::rethrow_exception(lastException);
//            }
//        }
//    }


};

#endif //GRAPH_SEARCH_UTIL_H
