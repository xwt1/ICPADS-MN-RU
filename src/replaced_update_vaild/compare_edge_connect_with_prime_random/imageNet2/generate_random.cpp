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


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <root_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];

    std::string data_path = root_path + "/data/imageNet/image.ds";
    std::string query_path = root_path + "/data/imageNet/image.q";
    std::string output_random_indice = root_path + "/data/imageNet2/imageNet_random_data.ivecs";

    std::vector<std::string> paths_to_create ={output_random_indice};
    util::create_directories(paths_to_create);

    int dim, num_data, num_queries;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_data);
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);

    size_t data_siz = data.size();

    //    limit:数字大小上限，size：一次生成多少个这样的数字，num：生成的次数

    int limit = data_siz;
    int size = 10000;
    int num = 200;

    try {
        // Generate random numbers
        auto random_numbers = util::generate_unique_random_numbers(limit, size, num);

        // Save to fvecs format
        util::save_to_ivecs(output_random_indice, random_numbers);

        std::cout << "Random numbers have been generated and saved in ivecs format." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}