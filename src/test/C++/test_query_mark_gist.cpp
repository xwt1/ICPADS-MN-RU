////
//// Created by root on 5/30/24.
////
//
//// 删除前20000个点,看看recall会怎么变
//#include <fstream>
//#include <vector>
//#include <random>
//#include <thread>
//#include <atomic>
//#include <mutex>
//#include "hnswlib/hnswlib.h"
//#include <numeric>
//#include <unordered_set>
//#include <algorithm>
//#include <unordered_map>
//#include <chrono>
//#include "util.h"
//
//int main(int argc, char* argv[]){
//    if (argc < 2) {
//        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
//        return 1;
//    }
//    std::string root_path = argv[1];
//
//    std::string data_path = root_path + "/data/gist/gist_base.fvecs";
//    std::string query_path = root_path + "/data/gist/gist_query.fvecs";
//
////    std::string index_path = root_path + "/data/gist/hnsw_maintenance/gist_hnsw_maintenance_index.bin";
//    std::string index_path = root_path + "/data/gist/hnsw_prime/gist_hnsw_prime_index.bin";
//
//    std::string ground_truth_path = root_path + "/data/gist/gist_groundtruth.ivecs";
////    std::string output_csv_path = root_path + "/output/figure_3/gist/direct_delete.csv";
//
////    std::string output_csv_path = root_path + "/output/figure_3/gist/direct_delete_method10.csv";
////    std::string output_index_path = root_path + "/output/figure_3/gist/method10_gist_after_mult_index.bin";
//
//
//    int dim, num_data, num_queries;
//    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
//    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
//
//    size_t data_siz = data.size();
//
//    int k = 100;
//
//    // Initialize the HNSW index
//    hnswlib::L2Space space(dim);
//    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
//
//    std::cout << "索引加载完毕 " << std::endl;
//    // 设置查询参数`ef`
//    int ef = 1000;
//    index.setEf(ef);
//
//    int num_threads = 48;
//
//    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
//
//    std::unordered_map<size_t, size_t> index_map;
//    for (size_t i = 0; i < num_data; ++i) {
//        index_map[i] = i;
//    }
//    std::vector<size_t> delete_indices;
//    for(int  i = 0 ; i< 100000;i++){
//        delete_indices.push_back(i);
//    }
//    util::markDeleteMultiThread(index, delete_indices, index_map, num_threads);
//    std::vector<std::vector<size_t>> labels;
//    util::query_hnsw(index, queries, k, num_threads, labels);
//    float recall = util::recall_score(ground_truth, labels, index_map, data_siz);
//    std::cout<<recall<<std::endl;
//    return  0;
//}


//// 这个地方是跑20次gist_mult,大概有25000个找不到的点,理论上Recall至少应该在97.5%左右
////
//// Created by root on 5/30/24.
////
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <random>
//#include <thread>
//#include <atomic>
//#include <mutex>
//#include "hnswlib/hnswlib.h"
//#include <numeric>
//#include <unordered_set>
//#include <algorithm>
//#include <unordered_map>
//#include <chrono>
//#include "util.h"
//
//int main(int argc, char* argv[]){
//    if (argc < 2) {
//        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
//        return 1;
//    }
//    std::string root_path = argv[1];
//
//    std::string data_path = root_path + "/data/gist/gist_base.fvecs";
//    std::string query_path = root_path + "/data/gist/gist_query.fvecs";
//    std::string index_path = root_path + "/data/gist/hnsw_prime/gist_hnsw_prime_index.bin";
//    std::string ground_truth_path = root_path + "/data/gist/gist_groundtruth.ivecs";
////    std::string output_csv_path = root_path + "/output/figure_3/gist/mark_delete_ef1000.csv";
//    std::string output_index_path = root_path + "/output/test/test_gist_unreachable.bin";
//
//
//    int dim, num_data, num_queries;
//    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
//    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
//
//    size_t data_siz = data.size();
//
//    int k = 100;
//
//    // Initialize the HNSW index
//    hnswlib::L2Space space(dim);
//    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
//    std::unordered_map<size_t, size_t> index_map;
//    for (size_t i = 0; i < num_data; ++i) {
//        index_map[i] = i;
//    }
//
//    std::cout << "索引加载完毕 " << std::endl;
//    // 设置查询参数`ef`
//    int ef = 100;
//    index.setEf(ef);
//    int num_threads = 48;
//
//
//    // Number of iterations for delete and re-add process
//    int num_iterations = 100;
//    std::random_device rd;
//    std::mt19937 gen(rd());
//
//    // Perform initial brute-force k-NN search to get ground truth
//    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
//
//
////    // generate CSV file
////    std::vector<std::vector <std::string>> header = {{"iteration_number" , "unreachable_points_number","recall","avg_delete_time",
////                                                      "avg_add_time","avg_sum_delete_add_time","avg_query_time"}};
////    util::writeCSVOut(output_csv_path, header);
//
//
//    size_t last_idx = 0;
//    for (int iteration = 0; iteration < 20; ++iteration) {
//        std::unordered_set<size_t> delete_indices_set;
//
//        // 计算后一半结点的起始下标
//        size_t start_idx = last_idx;
//        int num_to_delete = num_data / num_iterations;
//        double delete_rate = (double)num_to_delete / num_data;
//        last_idx =  start_idx+num_to_delete;
//
//
//        // 将后一半结点的下标插入到集合中
//        for (size_t idx = start_idx; idx < start_idx+num_to_delete; ++idx) {
//            delete_indices_set.insert(idx);
//        }
//
//        std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());
//
//
//        // Save the vectors and their labels to be deleted before deleting them
//        std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dim));
//        for (size_t i = 0; i < delete_indices.size(); ++i) {
//            size_t idx = delete_indices[i];
//            deleted_vectors[i] = data[idx];
//        }
//
//        auto start_time_delete = std::chrono::high_resolution_clock::now();
//        util::markDeleteMultiThread(index, delete_indices, index_map, num_threads);
//        auto end_time_delete = std::chrono::high_resolution_clock::now();
//        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();
//
//
//        // Re-add the deleted vectors with their original labels
//        std::vector<size_t> new_indices(delete_indices.size());
//        for (size_t i = 0; i < delete_indices.size(); ++i) {
//            size_t idx = index_map[delete_indices[i]];
//            size_t new_idx = (idx < num_data) ? idx + num_data : idx - num_data;
//            new_indices[i] = new_idx;
//            index_map[delete_indices[i]] = new_idx;
//        }
//
//        auto start_time_add = std::chrono::high_resolution_clock::now();
//        util::addPointsMultiThread(index, deleted_vectors, new_indices, num_threads);
//        auto end_time_add = std::chrono::high_resolution_clock::now();
//        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();
//
//        // Perform k-NN search and measure recall and query time
////        std::vector<std::vector<size_t>> labels;
//
////        auto start_time_query = std::chrono::high_resolution_clock::now();
//////        util::query_hnsw(index, queries, k, num_threads, labels);
////        auto end_time_query = std::chrono::high_resolution_clock::now();
////        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();
//
////        float recall = util::recall_score(ground_truth, labels, index_map, data_siz);
//
//        auto avg_delete_time = delete_duration / num_to_delete;
//        auto avg_add_time = add_duration / num_to_delete;
////        auto avg_query_time = query_duration / queries.size();
//
//        auto avg_sum_delete_add_time = avg_delete_time + avg_add_time;
//
//        std::cout << "------------------------------------------------------------------" << std::endl;
//        std::cout << "Iteration " << iteration + 1 << ":\n";
//        std::cout<<"删除了大约"<<delete_rate<<"的点"<<std::endl;
////        std::cout << "RECALL: " << recall << "\n";
//        std::cout << "Avg Delete Time: " << avg_delete_time << " seconds\n";
//        std::cout << "Avg Add Time: " << avg_add_time << " seconds\n";
////        std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
//        std::cout << "Avg SUM Delete Add Time: " << avg_sum_delete_add_time << " seconds\n";
//
//        std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+1);
//        auto results = util::query_index(&index, queries_tmp, data.size());
//        std::unordered_map <size_t,bool> excluded_global_labels_all;
//        for (size_t j = 0; j < queries_tmp.size(); ++j) {
//            std::cout << "Query " << j << ":" << std::endl;
//            std::cout << "Labels length: " << results[j].first.size() << ",只能找到这么多的点" << std::endl;
//        }
//        std::cout << "------------------------------------------------------------------" << std::endl;
//
//        std::string iteration_string = std::to_string(iteration + 1);
//        std::string unreachable_points_string = std::to_string(data_siz - results.front().first.size());
////        std::string recall_string = std::to_string(recall);
//        std::string avg_delete_time_string = std::to_string(avg_delete_time);
//        std::string avg_add_time_string = std::to_string(avg_add_time);
//        std::string avg_sum_delete_add_time_string = std::to_string(avg_sum_delete_add_time);
////        std::string avg_query_time_string = std::to_string(avg_query_time);
//
////        std::vector<std::vector <std::string>> result_data = {{iteration_string, unreachable_points_string,recall_string,
////                                                               avg_delete_time_string,avg_add_time_string,avg_sum_delete_add_time_string,avg_query_time_string}};
//
////        util::writeCSVApp(output_csv_path, result_data);
//    }
//    index.saveIndex(output_index_path);
//    return  0;
//}



//// 这个地方是跑20次gist_mult,大概有25000个找不到的点,理论上Recall至少应该在97.5%左右
////
//// Created by root on 5/30/24.
////
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <random>
//#include <thread>
//#include <atomic>
//#include <mutex>
//#include "hnswlib/hnswlib.h"
//#include <numeric>
//#include <unordered_set>
//#include <algorithm>
//#include <unordered_map>
//#include <chrono>
//#include <set>
//#include "util.h"
//
//int main(int argc, char* argv[]){
//    if (argc < 2) {
//        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
//        return 1;
//    }
//    std::string root_path = argv[1];
//
//    std::string data_path = root_path + "/data/gist/gist_base.fvecs";
//    std::string query_path = root_path + "/data/gist/gist_query.fvecs";
//    std::string index_path = root_path + "/data/gist/hnsw_prime/gist_hnsw_prime_index.bin";
//    std::string ground_truth_path = root_path + "/data/gist/gist_groundtruth.ivecs";
////    std::string output_csv_path = root_path + "/output/figure_3/gist/mark_delete_ef1000.csv";
//    std::string output_index_path = root_path + "/output/test/test_gist_unreachable.bin";
//
//
//    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
//
//    int dim, num_data, num_queries;
//    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
//    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
//
//    size_t data_siz = data.size();
//
//    int k = 100;
//    int ef = 1000;
//    int num_threads = 48;
//
//
//    std::unordered_map<size_t, size_t> index_map;
//    for (size_t i = 0; i < 200000; ++i) {
//        index_map[i] = i+data.size();
//    }
//    for (size_t i = 200000; i < data.size(); ++i) {
//        index_map[i] = i;
//    }
//
//    // Initialize the HNSW index
//    hnswlib::L2Space space(dim);
//    hnswlib::HierarchicalNSW<float> index(&space, output_index_path, false, data_siz, true);
//
//    index.setEf(ef);
//    std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+2);
//    auto results = util::query_index(&index, queries_tmp, data.size());
//    std::unordered_map <size_t,bool> have_labels;
//
//    for (size_t j = 0; j < queries_tmp.size(); ++j) {
//        std::cout << "Query " << j << ":" << std::endl;
//        std::cout << "Labels length: " << results[j].first.size() << ",只能找到这么多的点" << std::endl;
//        for(auto &item : results[j].first){
//            if(item >= data_siz){
//                have_labels[item - data_siz] = true;
//            }else{
//                have_labels[item] = true;
//            }
//        }
//    }
//    std::unordered_map <size_t, bool> doNotHave1;
//    std::vector <size_t> doNotHave;
//    // 收集0到一百万的数据
//    for(int i = 0 ; i < data_siz ; i++) {
//        if (have_labels.find(i) == have_labels.end()) {
//            doNotHave.push_back(i);
//            doNotHave1[i] = true;
//        }
//    }
//
//    auto last = std::unique(doNotHave.begin(), doNotHave.end());
//    // Erase the 'removed' elements
//    doNotHave.erase(last, doNotHave.end());
//
//    // Use a set to remove duplicates
//    std::set<int> uniqueSet(doNotHave.begin(), doNotHave.end());
//
//    // Copy the unique elements back to the vector
//    doNotHave.assign(uniqueSet.begin(), uniqueSet.end());
//
//    size_t mx = 0;
//    for(auto item: doNotHave){
//        mx = std::max(mx,item);
//    }
//    std::cout<<"mx :"<<mx<<std::endl;
//
//    std::cout<<"have_labels size "<<have_labels.size()<<std::endl;
//    std::cout<<"doNotHave1 size "<<doNotHave1.size()<<std::endl;
//    std::cout<<"doNotHave size "<<doNotHave.size()<<std::endl;
//    std::vector<std::vector<size_t>> labels;
//    util::query_hnsw(index, queries, k, num_threads, labels);
//    float recall = util::recall_score(ground_truth, labels, index_map, data_siz);
//    std::cout<<recall<<std::endl;
//
//    for(auto i: labels){
//        for(auto j : i){
//            int idx = (j >= data_siz)?j-data_siz:j;
//            if(doNotHave1.find(idx) != doNotHave1.end()){
//                std::cout<<"wtf,找不到的点被找到了"<<std::endl;
//            }
////            if()
//        }
//    }
//
//    int count = 0;
//    for(auto i:ground_truth){
//        for(auto j : i){
//            int idx = (j >= data_siz)?j-data_siz:j;
//            if(doNotHave1.find(idx) != doNotHave1.end()){
//                count++;
////                std::cout<<"wtf,找不到的点被找到了"<<std::endl;
//            }
////            if()
//        }
//    }
//
//
//    std::cout<<count<<std::endl;
//
//    //作为对比,看一下前25000个点在groundTruth中的占比
//    int count1 = 0;
//    for(auto i:ground_truth) {
//        for (auto j: i) {
//            if(0<=j && j <25000){
//                count1++;
//            }
//        }
//    }
//    std::cout<<count1<<std::endl;
//
//
//    const int numRandomNumbers = 25000;
//    const int lowerBound = 0;
//    const int upperBound = 1000000;
//
//    // Create a random number generator
//    std::random_device rd;  // Obtain a random seed from the hardware
//    std::mt19937 gen(rd()); // Seed the generator
//    std::uniform_int_distribution<> distr(lowerBound, upperBound); // Define the range
//
//    // Generate unique random numbers
//    std::unordered_set<int> uniqueRandomNumbers;
//    while (uniqueRandomNumbers.size() < numRandomNumbers) {
//        uniqueRandomNumbers.insert(distr(gen));
//    }
//
//    int count2 = 0;
//    for(auto i:ground_truth) {
//        for (auto j: i) {
//            if(uniqueRandomNumbers.find(j) != uniqueRandomNumbers.end()){
//                count2++;
//            }
//        }
//    }
//    std::cout<<count2<<std::endl;
//
//
////    hnswlib::HierarchicalNSW<float> index2(&space, index_path, false, data_siz, true);
////    index2.setEf(ef);
////    std::vector<size_t> delete_indices;
////    for(auto j:doNotHave){
////        delete_indices.push_back(j);
////    }
////
////    std::unordered_map<size_t, size_t> index_map2;
////    for (size_t i = 0; i < data_siz; ++i) {
////        index_map2[i] = i;
////    }
////
////    util::markDeleteMultiThread(index2, delete_indices, index_map2, num_threads);
////    std::vector<std::vector<size_t>> label2;
////    util::query_hnsw(index2, queries, k, num_threads, label2);
////    float recall2 = util::recall_score(ground_truth, label2, index_map2, data_siz);
////    std::cout<<recall2<<std::endl;
//
//
//
//
//
//
//
////    std::cout<<"___________________________________"<<std::endl;
////    for(auto i: doNotHave){
////        std::cout<<i<<std::endl;
////    }
//
////    std::vector <int> wtf = {1,1,1,1,1};
////    auto last = std::unique(wtf.begin(), wtf.end());
////
////    // Erase the 'removed' elements
////    wtf.erase(last, wtf.end());
////    std::cout<<wtf.size()<<std::endl;
//
//    return  0;
//}


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
#include "util.h"

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/data/netflix/netflix_base.fvecs";
    std::string query_path = root_path + "/data/netflix/netflix_query_all.fvecs";
    std::string index_path = root_path + "/data/netflix/hnsw_prime/netflix_hnsw_prime_index.bin";
    std::string ground_truth_path = root_path + "/data/netflix/netflix_groundtruth.ivecs";
    std::string output_csv_path = root_path + "/output/full_coverage/netflix/replaced_update.csv";
    std::string output_index_path = root_path + "/output/full_coverage/netflix/replaced_update_netflix_full_coverage_index.bin";

    std::vector<std::string> paths_to_create ={output_csv_path,output_index_path};
    util::create_directories(paths_to_create);

    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);

    size_t data_siz = data.size();

    int k = 5;

    // Initialize the HNSW index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, output_index_path, false, data_siz, true);
    std::unordered_map<size_t, size_t> index_map;
    for (size_t i = 0; i < num_data; ++i) {
        index_map[i] = i;
    }

    std::cout << "索引加载完毕 " << std::endl;
    // 设置查询参数`ef`
    int ef = 500;
    index.setEf(ef);

    int num_threads = 40;

    // Number of iterations for delete and re-add process
    int num_iterations = 100;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Perform initial brute-force k-NN search to get ground truth
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);

    // Perform k-NN search and measure recall and query time
    std::vector<std::vector<size_t>> labels;

    auto start_time_query = std::chrono::high_resolution_clock::now();
    util::query_hnsw(index, queries, k, num_threads, labels);
    auto end_time_query = std::chrono::high_resolution_clock::now();


    auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();
    auto avg_query_time = query_duration / queries.size();
    float recall = util::recall_score(ground_truth, labels, index_map, data_siz);

    std::cout << "RECALL: " << recall << "\n";
    std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
    return  0;
}


