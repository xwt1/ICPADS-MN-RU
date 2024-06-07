//
// Created by root on 5/30/24.
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

// Load fvecs file format into 2D vector
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

std::vector<std::vector<size_t>> load_ivecs_indices(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<size_t>> indices;
    while (input.peek() != EOF) {
        int dim;
        input.read(reinterpret_cast<char*>(&dim), sizeof(int));

        std::vector<int> vec(dim);
        input.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        indices.emplace_back(vec.begin(), vec.end());
    }

    input.close();
    return indices;
}

float recall_score(const std::vector<std::vector<size_t>>& ground_truth, const std::vector<std::vector<size_t>>& predictions, const std::unordered_map<size_t, size_t>& index_map, size_t data_size) {
    size_t hit_count = 0;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        std::unordered_set<size_t> true_set(ground_truth[i].begin(), ground_truth[i].end());
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            size_t predicted_index = predictions[i][j];
            if (predicted_index >= data_size) {
                predicted_index = predicted_index - data_size;
            }
            if (true_set.find(predicted_index) != true_set.end()) {
                hit_count++;
            }
        }
    }
    return static_cast<float>(hit_count) / (ground_truth.size() * ground_truth[0].size());
}

std::vector<std::vector<size_t>> brute_force_knn(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& queries, int dim, int k) {
    size_t num_data = data.size();
    size_t num_queries = queries.size();
    std::vector<std::vector<size_t>> indices(num_queries, std::vector<size_t>(k));

    for (size_t i = 0; i < num_queries; ++i) {
        std::vector<std::pair<float, size_t>> distances(num_data);
        for (size_t j = 0; j < num_data; ++j) {
            float dist = 0;
            for (int d = 0; d < dim; ++d) {
                float diff = data[j][d] - queries[i][d];
                dist += diff * diff;
            }
            distances[j] = { dist, j };
        }
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        for (int n = 0; n < k; ++n) {
            indices[i][n] = distances[n].second;
        }
    }

    return indices;
}

void query_hnsw(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, int num_threads, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times) {
    size_t num_queries = queries.size();
    labels.resize(num_queries, std::vector<size_t>(k));
    query_times.resize(num_queries);

    auto query_func = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto result = index.searchKnn(queries[i].data(), k);
            auto t2 = std::chrono::high_resolution_clock::now();
            query_times[i] = std::chrono::duration<double>(t2 - t1).count();
            for (size_t j = 0; j < k; ++j) {
                labels[i][j] = result.top().second;
                result.pop();
            }
        }
    };

    std::vector<std::thread> threads;
    size_t step = num_queries / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        size_t start = i * step;
        size_t end = (i == num_threads - 1) ? num_queries : start + step;
        threads.emplace_back(query_func, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void query_hnsw_single(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times) {
    size_t num_queries = queries.size();
    labels.resize(num_queries, std::vector<size_t>(k));
    query_times.resize(num_queries);

    for (size_t i = 0; i < num_queries; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = index.searchKnn(queries[i].data(), k);
        auto t2 = std::chrono::high_resolution_clock::now();
        query_times[i] = std::chrono::duration<double>(t2 - t1).count();
        for (size_t j = 0; j < k; ++j) {
            labels[i][j] = result.top().second;
            result.pop();
        }
    }
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

void DirectDeleteMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<size_t>& delete_indices, const std::unordered_map<size_t, size_t>& index_map, int num_threads) {
    size_t num_delete = delete_indices.size();

    ParallelFor(0, num_delete, num_threads, [&](size_t i, size_t) {
        size_t idx = index_map.at(delete_indices[i]);
        index.directDelete(idx);
    });
}

void addPointsMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& points, const std::vector<size_t>& labels, int num_threads) {
    size_t num_points = points.size();

    ParallelFor(0, num_points, num_threads, [&](size_t i, size_t) {
        index.addPoint(points[i].data(), labels[i], true);
    });
}

int main(){
    std::string data_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
    std::string query_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_query.fvecs";

//    std::string index_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/freshdiskann_prove/diskann_prove.bin";
    std::string index_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/direct_delete/direct_delete.bin";

    std::string ground_truth_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/sift/sift_groundtruth.ivecs";

    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = load_fvecs(query_path, dim, num_queries);

    size_t data_siz = data.size();

    int k = 100;
    int num_threads = std::thread::hardware_concurrency();

    // Initialize the HNSW index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    std::cout << "索引加载完毕 " << std::endl;
    // 设置查询参数`ef`
    int ef = 500;
    index.setEf(ef);

    // Number of iterations for delete and re-add process
    int num_iterations = 10;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Perform initial brute-force k-NN search to get ground truth
    std::vector<std::vector<size_t>> ground_truth = load_ivecs_indices(ground_truth_path);

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
//        // Select the indices to delete
//        int num_to_delete = num_data * 0.01;
//        std::unordered_set<size_t> delete_indices_set;
//        while (delete_indices_set.size() < num_to_delete) {
//            size_t idx = std::uniform_int_distribution<size_t>(0, num_data - 1)(gen);
//            delete_indices_set.insert(idx);
//        }
//        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());
        std::unordered_set<size_t> delete_indices_set;

        // 计算后一半结点的起始下标
        size_t start_idx = 0;
        double delete_rate = 0.1;
        int num_to_delete = num_data * delete_rate;
//        int num_to_delete = 5;

        std::cout<<"删除了大约"<<delete_rate<<"的点"<<std::endl;
        // 将后一半结点的下标插入到集合中
        for (size_t idx = start_idx; idx < start_idx+num_to_delete; ++idx) {
            delete_indices_set.insert(idx);
//            // 如果集合大小达到需要删除的数量，则跳出循环
//            if (delete_indices_set.size() >= num_to_delete) {
//                break;
//            }
        }

        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());


        // Save the vectors and their labels to be deleted before deleting them
        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = delete_indices[i];
            deleted_vectors[i] = data[idx];
        }
        index.checkTotalInOutDegreeEquality();

        std::cout<<"开始删除"<<std::endl;

        auto start_time_delete = std::chrono::high_resolution_clock::now();

//        for(int i = 0 ; i < delete_indices.size();i++){
//            int delete_id = index_map[delete_indices[i]];
//            index.directDelete(delete_id);
//        }
//        std::cout<<"删除完成"<<std::endl;

//        for(int i = 0 ; i < delete_indices.size();i++){
//            int delete_id = index_map[delete_indices[i]];
//            index.printInDegreeVertices(delete_id);
//        }

//
        DirectDeleteMultiThread(index, delete_indices, index_map, 1);
        std::cout<<"删除完成"<<std::endl;
        index.checkTotalInOutDegreeEquality();

//        for(int i = 0 ; i < delete_indices.size();i++){
//            int delete_id = index_map[delete_indices[i]];
//            index.printInDegreeVertices(delete_id);
//        }

        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

        // Re-add the deleted vectors with their original labels
        std::vector<size_t> new_indices(delete_indices.size());
        for (size_t i = 0; i < delete_indices.size(); ++i) {
            size_t idx = index_map[delete_indices[i]];
            size_t new_idx = (idx < num_data) ? idx + num_data : idx - num_data;
            new_indices[i] = new_idx;
            index_map[delete_indices[i]] = new_idx;
        }

        auto start_time_add = std::chrono::high_resolution_clock::now();
        addPointsMultiThread(index, deleted_vectors, new_indices, 1);
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();



        // Perform k-NN search and measure recall and query time
        std::vector<std::vector<size_t>> labels;
        std::vector<double> query_times;

//        query_hnsw(index, queries, dim, k, 1, labels, query_times);
        query_hnsw_single(index, queries, dim, k, labels, query_times);

        float recall = recall_score(ground_truth, labels, index_map, data_siz);
        double avg_query_time = std::accumulate(query_times.begin(), query_times.end(), 0.0) / query_times.size();

        double sum_time = std::accumulate(query_times.begin(), query_times.end(), 0.0);
        sum_time += add_duration;
        sum_time += delete_duration;

        std::cout << "Iteration " << iteration + 1 << ":\n";
        std::cout << "RECALL: " << recall << "\n";
        std::cout << "Query Time: " << avg_query_time << " seconds\n";
        std::cout << "Delete Time: " << delete_duration << " seconds\n";
        std::cout << "Add Time: " << add_duration << " seconds\n";
        std::cout << "SUM Time: " << sum_time << " seconds\n";

//        std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+5);
//        auto results = query_index(&index, queries_tmp, 1000000);
//        std::unordered_map <size_t,bool> excluded_global_labels_all;
//        for (size_t j = 0; j < queries_tmp.size(); ++j) {
//            std::cout << "Query " << j << ":" << std::endl;
//            std::cout << "Labels length: " << results[j].first.size() << ",只能找到这么多的点" << std::endl;
//        }

   }
}