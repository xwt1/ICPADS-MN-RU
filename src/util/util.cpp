//
// Created by root on 6/6/24.
//

#include "util/util.h"

std::vector<std::vector<float>> util::load_fvecs(const std::string &filename, int &dim, int &num) {
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

std::vector<std::vector<size_t>> util::load_ivecs_indices(const std::string& filename) {
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

void util::query_hnsw(hnswlib::HierarchicalNSW<float>& alg_hnsw, const std::vector<std::vector<float>>& queries, int k, int num_threads, std::vector<std::vector<size_t>>& results) {
    size_t num_queries = queries.size();
    results.resize(num_queries, std::vector<size_t>(k));
    ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw.searchKnn(queries[row].data(), k);
        std::vector<size_t> neighbors;
        while (!result.empty()) {
            neighbors.push_back(result.top().second);
            result.pop();
        }
        std::reverse(neighbors.begin(), neighbors.end()); // reverse to get correct order
        results[row] = neighbors;
    });
}

//void util::query_hnsw(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, int num_threads, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times) {
//    size_t num_queries = queries.size();
//    labels.resize(num_queries, std::vector<size_t>(k));
//    query_times.resize(num_queries);
//
//    auto query_func = [&](size_t start, size_t end) {
//        for (size_t i = start; i < end; ++i) {
//            auto t1 = std::chrono::high_resolution_clock::now();
//            auto result = index.searchKnn(queries[i].data(), k);
//            auto t2 = std::chrono::high_resolution_clock::now();
//            query_times[i] = std::chrono::duration<double>(t2 - t1).count();
//            for (size_t j = 0; j < k; ++j) {
//                labels[i][j] = result.top().second;
//                result.pop();
//            }
//        }
//    };
//
//    std::vector<std::thread> threads;
//    size_t step = num_queries / num_threads;
//    for (int i = 0; i < num_threads; ++i) {
//        size_t start = i * step;
//        size_t end = (i == num_threads - 1) ? num_queries : start + step;
//        threads.emplace_back(query_func, start, end);
//    }
//
//    for (auto& thread : threads) {
//        thread.join();
//    }
//}

void util::query_hnsw_single(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times) {
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

std::vector<std::pair<std::vector<size_t>, std::vector<float>>> util::query_index(hnswlib::HierarchicalNSW<float>* index, const std::vector<std::vector<float>> &queries, int k) {
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

void util::markDeleteMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<size_t>& delete_indices, const std::unordered_map<size_t, size_t>& index_map, int num_threads) {
    size_t num_delete = delete_indices.size();

    ParallelFor(0, num_delete, num_threads, [&](size_t i, size_t) {
        size_t idx = index_map.at(delete_indices[i]);
        index.markDelete(idx);
    });
}

void util::addPointsMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& points, const std::vector<size_t>& labels, int num_threads) {
    size_t num_points = points.size();

    ParallelFor(0, num_points, num_threads, [&](size_t i, size_t) {
        index.addPoint(points[i].data(), labels[i], true);
    });
}


void util::writeCSVOut(const std::string& filename, const std::vector<std::vector<std::string>>& data) {
    std::filesystem::path file_path(filename);
    std::filesystem::path dir_path = file_path.parent_path();

    // Check if directory exists, if not, create it
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }

    std::ofstream file;
    file.open(filename, std::ios_base::out);
//    if (std::filesystem::exists(file_path)) {
//        file.open(filename, std::ios_base::app); // Open file in append mode if it exists
//    } else {
//        file.open(filename, std::ios_base::out); // Create a new file if it does not exist
//    }

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    for (const auto& row : data) {
        std::stringstream ss;
        for (size_t i = 0; i < row.size(); ++i) {
            ss << row[i];
            if (i < row.size() - 1) {
                ss << ",";
            }
        }
        file << ss.str() << "\n";
    }

    file.close();
}

void util::writeCSVApp(const std::string& filename, const std::vector<std::vector<std::string>>& data) {
    std::filesystem::path file_path(filename);
    std::filesystem::path dir_path = file_path.parent_path();

    // Check if directory exists, if not, create it
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }

    std::ofstream file;
    file.open(filename, std::ios_base::app);
//    if (std::filesystem::exists(file_path)) {
//        file.open(filename, std::ios_base::app); // Open file in append mode if it exists
//    } else {
//        file.open(filename, std::ios_base::out); // Create a new file if it does not exist
//    }

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    for (const auto& row : data) {
        std::stringstream ss;
        for (size_t i = 0; i < row.size(); ++i) {
            ss << row[i];
            if (i < row.size() - 1) {
                ss << ",";
            }
        }
        file << ss.str() << "\n";
    }

    file.close();
}

float util::recall_score(const std::vector<std::vector<size_t>>& ground_truth, const std::vector<std::vector<size_t>>& predictions, const std::unordered_map<size_t, size_t>& index_map, size_t data_size) {
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

void util::knn_thread(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& queries, int dim, int k, size_t start_idx, size_t end_idx, std::vector<std::vector<size_t>>& indices) {
    size_t num_data = data.size();

    for (size_t i = start_idx; i < end_idx; ++i) {
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
}

std::vector<std::vector<size_t>> util::brute_force_knn(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& queries, int dim, int k) {
    size_t num_data = data.size();
    size_t num_queries = queries.size();
    std::vector<std::vector<size_t>> indices(num_queries, std::vector<size_t>(k));

    auto knn_thread = [&](size_t start_idx, size_t end_idx) {
        for (size_t i = start_idx; i < end_idx; ++i) {
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
    };

    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = (num_queries + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, num_queries);
        if (start_idx < end_idx) {
            threads.emplace_back(knn_thread, start_idx, end_idx);
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return indices;
}

void util::save_knn_to_ivecs(const std::string& filename, const std::vector<std::vector<size_t>>& knn_results) {
    std::ofstream output_file(filename, std::ios::binary);
    if (!output_file) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    for (const auto& neighbors : knn_results) {
        int k = neighbors.size();
        output_file.write(reinterpret_cast<const char*>(&k), sizeof(k));
        for (size_t neighbor : neighbors) {
            int id = static_cast<int>(neighbor);
            output_file.write(reinterpret_cast<const char*>(&id), sizeof(id));
        }
    }
}



void util::create_directories(const std::vector<std::string>& paths) {
    for (const auto& path_str : paths) {
        std::filesystem::path path(path_str);
        std::filesystem::path directory_path;

        if (path.has_extension()) {
            directory_path = path.parent_path();
        } else {
            directory_path = path;
        }

        try {
            if (!std::filesystem::exists(directory_path)) {
                std::filesystem::create_directories(directory_path);
                std::cout << "Directories created: " << directory_path << std::endl;
            } else {
                std::cout << "Path already exists: " << directory_path << std::endl;
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Filesystem error creating " << directory_path << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating " << directory_path << ": " << e.what() << std::endl;
        }
    }
}

std::vector<std::vector<size_t>> util::generate_unique_random_numbers(int limit, int size, int num) {
    if (size > limit) {
        throw std::invalid_argument("size cannot be greater than limit + 1");
    }

    std::vector<std::vector<size_t>> results;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int n = 0; n < num; ++n) {
        std::vector<size_t> available_numbers(limit);
        std::iota(available_numbers.begin(), available_numbers.end(), 0); // Fill with 0, 1, ..., limit -1

        std::vector<size_t> random_numbers(size);
        for (int i = 0; i < size; ++i) {
            std::uniform_int_distribution<> dis(0, available_numbers.size() - 1);
            int index = dis(gen);
            random_numbers[i] = available_numbers[index];
            available_numbers.erase(available_numbers.begin() + index);
        }
        results.push_back(random_numbers);
    }

    return results;
}

void util::save_to_fvecs(const std::string& filename, const std::vector<std::vector<int>>& data) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    for (const auto& numbers : data) {
        int size = numbers.size();
        outfile.write(reinterpret_cast<const char*>(&size), sizeof(int)); // Write the size
        outfile.write(reinterpret_cast<const char*>(numbers.data()), size * sizeof(int)); // Write the numbers
    }

    outfile.close();
}

void util::save_to_ivecs(const std::string& filename, const std::vector<std::vector<size_t>>& data) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("无法打开文件进行写入");
    }

    for (const auto& numbers : data) {
        int size = static_cast<int>(numbers.size());
        outfile.write(reinterpret_cast<const char*>(&size), sizeof(int)); // 写入维度
        outfile.write(reinterpret_cast<const char*>(numbers.data()), size * sizeof(size_t)); // 写入数据
    }

    outfile.close();
}


std::vector<std::vector<size_t>> util::load_ivecs(const std::string &filename, int &dim, int &num) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<size_t>> data;
    num = 0;

    while (input.peek() != EOF) {
        int current_dim;
        input.read(reinterpret_cast<char*>(&current_dim), sizeof(int)); // 读取每个向量的维度信息
        if (num == 0) {
            dim = current_dim; // 记录第一个向量的维度
        }
        std::vector<size_t> temp(current_dim);
        input.read(reinterpret_cast<char*>(temp.data()), current_dim * sizeof(size_t)); // 读取数据
        data.push_back(std::move(temp));
        num++;
    }

    input.close();
    return data;
}


//std::string util::readJsonFile(const std::string& filename, const std::string& key) {
//    std::ifstream ifs(filename);
//    if (!ifs.is_open()) {
//        std::cerr << "Could not open the file: " << filename << std::endl;
//        return "";
//    }
//
//    std::string line, jsonContent, result;
//    while (std::getline(ifs, line)) {
//        jsonContent += line;
//    }
//
//    size_t keyPos = jsonContent.find("\"" + key + "\"");
//    if (keyPos != std::string::npos) {
//        size_t colonPos = jsonContent.find(":", keyPos);
//        size_t startQuote = jsonContent.find("\"", colonPos);
//        size_t endQuote = jsonContent.find("\"", startQuote + 1);
//        if (startQuote != std::string::npos && endQuote != std::string::npos) {
//            result = jsonContent.substr(startQuote + 1, endQuote - startQuote - 1);
//        }
//    }
//
//    ifs.close();
//    return result;
//}