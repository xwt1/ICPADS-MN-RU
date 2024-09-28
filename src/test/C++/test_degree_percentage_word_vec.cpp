
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/data/word2vec/word2vec_base.fvecs";
    std::string query_path = root_path + "/data/word2vec/word2vec_query.fvecs";
    std::string index_path = root_path + "/data/word2vec/hnsw_prime/word2vec_hnsw_prime_index.bin";
//    std::string ground_truth_path = root_path + "/data/msong/imageNet_groundtruth.ivecs";

    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);

    std::cout<<queries.size()<<std::endl;

    size_t data_siz = data.size();

    int k = 100;

    // Initialize the HNSW index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, data_siz, true);

    index.computeReciprocalNeighborRatio_out();
    return 0;
}