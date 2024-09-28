////
//// Created by root on 5/30/24.
////
//
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
//#include "direct_delete_util.h"
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
//    std::vector<std::vector<float>> data = directDeleteUtil::load_fvecs(data_path, dim, num_data);
//    std::vector<std::vector<float>> queries = directDeleteUtil::load_fvecs(query_path, dim, num_queries);
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
//    int num_threads = 8;
//
//    std::vector<std::vector<size_t>> ground_truth = directDeleteUtil::load_ivecs_indices(ground_truth_path);
//
//    std::unordered_map<size_t, size_t> index_map;
//    for (size_t i = 0; i < num_data; ++i) {
//        index_map[i] = i;
//    }
//    std::vector<size_t> delete_indices;
//    for(int  i = 0 ; i< 30000;i++){
//        delete_indices.push_back(i);
//    }
//    directDeleteUtil::directDeleteMultiThread(index, delete_indices, index_map, 1);
//    std::vector<std::vector<size_t>> labels;
//    directDeleteUtil::query_hnsw(index, queries, k, 1, labels);
//    float recall = directDeleteUtil::recall_score(ground_truth, labels, index_map, data_siz);
//    std::cout<<recall<<std::endl;
//    return  0;
//}



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
//#include "direct_delete_util.h"
//
//int main(int argc, char* argv[]){
////    if (argc < 2) {
////        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
////        return 1;
////    }
////    std::string root_path = argv[1];
//    std::string query_path = "/root/dataset/netflix/netflix_query_all.fvecs";
//    int dim, num_data, num_queries;
//    std::vector<std::vector<float>> queries = directDeleteUtil::load_fvecs(query_path, dim, num_queries);
//    std::cout<<queries.size() / 300 <<std::endl;
//    return 0;
//}


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
//    std::string data_path = root_path + "/data/netflix/netflix_base.fvecs";
//    std::string query_path = root_path + "/data/netflix/netflix_query_all.fvecs";
//    std::string index_path = root_path + "/data/netflix/hnsw_prime/netflix_hnsw_prime_index.bin";
//    std::string ground_truth_path = root_path + "/data/netflix/netflix_groundtruth.ivecs";
//    std::string output_csv_path = root_path + "/output/full_coverage/netflix/replaced_update.csv";
//    std::string output_index_path = root_path + "/output/full_coverage/netflix/edge_connected_replaced_update9_netflix_full_coverage_index.bin";
//
//    std::vector<std::string> paths_to_create ={output_csv_path,output_index_path};
//    util::create_directories(paths_to_create);
//
//    int dim, num_data, num_queries;
//    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
//    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
//
//    size_t data_siz = data.size();
//
//    int k = 5;
//
//    // Initialize the HNSW index
//    hnswlib::L2Space space(dim);
//    hnswlib::HierarchicalNSW<float> index(&space, output_index_path, false, data_siz, true);
//    std::unordered_map<size_t, size_t> index_map;
//    for (size_t i = 0; i < num_data; ++i) {
//        index_map[i] = i;
//    }
//
//    std::cout << "索引加载完毕 " << std::endl;
//    // 设置查询参数`ef`
//    int ef = 500;
//    index.setEf(ef);
//
//    int num_threads = 40;
//
//    // Number of iterations for delete and re-add process
//    int num_iterations = 100;
//    std::random_device rd;
//    std::mt19937 gen(rd());
//
//    // Perform initial brute-force k-NN search to get ground truth
//    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
//
//    // Perform k-NN search and measure recall and query time
//    std::vector<std::vector<size_t>> labels;
//
//    auto start_time_query = std::chrono::high_resolution_clock::now();
//    util::query_hnsw(index, queries, k, num_threads, labels);
//    auto end_time_query = std::chrono::high_resolution_clock::now();
//
//
//    auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();
//    auto avg_query_time = query_duration / queries.size();
//    float recall = util::recall_score(ground_truth, labels, index_map, data_siz);
//
//    std::cout << "RECALL: " << recall << "\n";
//    std::cout << "Avg Query Time: " << avg_query_time << " seconds\n";
//    return  0;
//}


//
// Created by root on 6/22/24.
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
//    std::string data_path = root_path + "/data/gist/gist_base.fvecs";
//    std::string query_path = root_path + "/data/gist/gist_query.fvecs";
//    std::string ground_truth_path = root_path + "/data/gist/gist_groundtruth.ivecs";



    std::vector <std::vector <std::string>> csv_index_path_vec={
            {root_path + "/output/full_coverage/word2vec/prime_word2vec.csv",
                    root_path + "/data/word2vec/hnsw_prime/word2vec_hnsw_prime_index.bin",
                    root_path + "/data/word2vec/word2vec_base.fvecs",
                    root_path + "/data/word2vec/word2vec_query.fvecs",
                    root_path + "/data/word2vec/word2vec_groundtruth.ivecs"}
    };

    for(auto csv_index : csv_index_path_vec){
        auto csv_path = csv_index[0];
        auto index_path = csv_index[1];
        auto data_path = csv_index[2];
        auto query_path = csv_index[3];
        auto ground_truth_path = csv_index[4];


        int dim, num_data, num_queries;
        std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
        std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);
        size_t data_siz = data.size();
        int k = 100;
        if(data_siz == 2340373){
            k = 10;
        }
        std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);
        std::unordered_map<size_t, size_t> index_map;
        for (size_t i = 0; i < num_data; ++i) {
            index_map[i + data_siz] = i;
        }


        // Initialize the HNSW index
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);
        std::cout << "索引加载完毕 " << std::endl;

        int start_ef = 1000;
        int end_ef = 10000;
        int step = 100;
        int num_threads = 40;





        std::vector<std::vector<double>> recall_time_vector = util::countRecallWithDiffPara(index,queries,ground_truth,index_map,k,start_ef,end_ef,step,num_threads,data_siz);

        // generate CSV file
        std::vector<std::vector <std::string>> header = {{"ef" , "recall", "query_time"}};
        util::writeCSVOut(csv_path, header);

        for(auto recall_time : recall_time_vector){

            std::string ef_string = std::to_string(start_ef);
            std::string recall_string = std::to_string(recall_time[0]);
            std::string query_time_string = std::to_string(recall_time[1]);
            std::vector<std::vector <std::string>> result_data = {{ef_string, recall_string,query_time_string}};
            util::writeCSVApp(csv_path, result_data);
            start_ef += step;
        }
    }

    return 0;
}