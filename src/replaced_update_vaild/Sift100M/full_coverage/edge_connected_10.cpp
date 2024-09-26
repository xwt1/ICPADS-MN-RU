#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include "util.h"  // 需要确保 util.h 中包含必要的函数声明

// 添加 get_fvecs_file_info 函数
void get_fvecs_file_info(const std::string& filename, int& dim, size_t& num_vectors) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    // 读取第一个向量的维度
    int temp_dim;
    input.read(reinterpret_cast<char*>(&temp_dim), sizeof(int));
    if (input.fail()) {
        std::cerr << "读取维度失败。" << std::endl;
        exit(1);
    }
    dim = temp_dim;

    // 获取文件大小
    input.seekg(0, std::ios::end);
    std::streampos file_size = input.tellg();

    // 计算向量数量
    size_t vector_size_in_bytes = sizeof(int) + sizeof(float) * dim;
    num_vectors = file_size / vector_size_in_bytes;

    input.close();
}

// 添加 load_fvecs_range 函数
std::vector<std::vector<float>> load_fvecs_range(const std::string& filename, size_t start_idx, size_t count, int dim) {
    std::vector<std::vector<float>> data;
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }

    size_t vector_size_in_bytes = sizeof(int) + sizeof(float) * dim;
    size_t start_pos = start_idx * vector_size_in_bytes;

    // 移动文件指针到起始位置
    input.seekg(start_pos, std::ios::beg);
    if (input.fail()) {
        std::cerr << "无法定位到文件中的起始位置。" << std::endl;
        exit(1);
    }

    for (size_t i = 0; i < count; ++i) {
        int cur_dim = 0;
        input.read(reinterpret_cast<char*>(&cur_dim), sizeof(int));
        if (cur_dim != dim) {
            std::cerr << "维度不匹配，预期: " << dim << ", 实际: " << cur_dim << std::endl;
            exit(1);
        }
        std::vector<float> vec(dim);
        input.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * dim);
        if (input.fail()) {
            std::cerr << "读取向量失败。" << std::endl;
            exit(1);
        }
        data.push_back(std::move(vec));
    }

    input.close();
    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <root_path> <output_path>" << std::endl;
        return 1;
    }
    std::string root_path = argv[1];  // 例如 "/root/WorkSpace/dataset"
    std::string output_path = argv[2];

    std::string data_path = root_path + "/sift/bigann_base.bvecs";
    std::string query_path = root_path + "/sift/sift200M/bigann_query.bvecs";
    std::string index_path = root_path + "/sift/sift200M/index/sift_100M_index.bin";
    std::string ground_truth_path = root_path + "/sift/sift200M/gnd/idx_100M.ivecs";
    std::string output_csv_path = output_path + "/output/full_coverage/sift200M/edge_connected_replaced_update10.csv";
    std::string output_index_path = output_path + "/output/full_coverage/sift200M/edge_connected_replaced_update10_sift100M_full_coverage_index.bin";

    // 确保输出目录存在
    std::vector<std::string> paths_to_create = {output_csv_path, output_index_path};
    util::create_directories(paths_to_create);

    // 加载查询数据
    int dim = 0, num_queries = 0;
    std::vector<std::vector<float>> queries = util::load_fvecs(query_path, dim, num_queries);

    // 获取数据集信息而不加载到内存中
    size_t num_data = 0;
    get_fvecs_file_info(data_path, dim, num_data);

    int k = 100;  // 最近邻数量

    // 初始化 HNSW 索引
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, index_path, false, num_data, true);

    std::cout << "索引加载完毕。" << std::endl;

    // 设置查询参数 'ef'
    int ef = 500;
    index.setEf(ef);

    int num_threads = 40;
    int num_iterations = 100;
    size_t num_to_delete = num_data / num_iterations;

    // 加载真值（ground truth）
    std::vector<std::vector<size_t>> ground_truth = util::load_ivecs_indices(ground_truth_path);

    // 准备 CSV 输出
    std::vector<std::vector<std::string>> header = {{"iteration_number", "unreachable_points_number", "recall", "avg_delete_time",
                                                     "avg_add_time", "avg_sum_delete_add_time", "avg_query_time"}};
    util::writeCSVOut(output_csv_path, header);

    // 用于跟踪重新添加点的标签映射
    std::unordered_map<size_t, size_t> index_map;

    size_t data_siz = num_data;  // 原始数据集大小

    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        size_t start_idx = iteration * num_to_delete;

        // 生成要删除的索引
        std::vector<size_t> delete_indices(num_to_delete);
        for (size_t i = 0; i < num_to_delete; ++i) {
            delete_indices[i] = start_idx + i;
        }

        // 加载要删除的向量
        std::vector<std::vector<float>> deleted_vectors = load_fvecs_range(data_path, start_idx, num_to_delete, dim);

        // 从索引中删除点
        auto start_time_delete = std::chrono::high_resolution_clock::now();
        util::markDeleteMultiThread(index, delete_indices, index_map, num_threads);
        auto end_time_delete = std::chrono::high_resolution_clock::now();
        auto delete_duration = std::chrono::duration<double>(end_time_delete - start_time_delete).count();

        // 使用新标签重新添加被删除的向量
        std::vector<size_t> new_indices(num_to_delete);
        for (size_t i = 0; i < num_to_delete; ++i) {
            size_t old_idx = delete_indices[i];
            size_t new_idx = (old_idx < num_data) ? old_idx + num_data : old_idx - num_data;
            new_indices[i] = new_idx;
            index_map[old_idx] = new_idx;  // 更新标签映射
        }

        // 将点重新添加到索引中
        auto start_time_add = std::chrono::high_resolution_clock::now();
        util::addPointsMultiThread(index, deleted_vectors, new_indices, num_threads);
        auto end_time_add = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration<double>(end_time_add - start_time_add).count();

        // 执行 k-NN 搜索并测量召回率和查询时间
        std::vector<std::vector<size_t>> labels;

        auto start_time_query = std::chrono::high_resolution_clock::now();
        util::query_hnsw(index, queries, k, num_threads, labels);
        auto end_time_query = std::chrono::high_resolution_clock::now();
        auto query_duration = std::chrono::duration<double>(end_time_query - start_time_query).count();

        // 计算召回率
        float recall = util::recall_score(ground_truth, labels, index_map, data_siz);

        double avg_delete_time = delete_duration / num_to_delete;
        double avg_add_time = add_duration / num_to_delete;
        double avg_query_time = query_duration / queries.size();
        double avg_sum_delete_add_time = avg_delete_time + avg_add_time;

        // 输出迭代结果
        std::cout << "------------------------------------------------------------------" << std::endl;
        std::cout << "第 " << iteration + 1 << " 次迭代:\n";
        std::cout << "删除了大约 " << ((double)num_to_delete / num_data) << " 的点" << std::endl;
        std::cout << "召回率: " << recall << "\n";
        std::cout << "平均删除时间: " << avg_delete_time << " 秒\n";
        std::cout << "平均添加时间: " << avg_add_time << " 秒\n";
        std::cout << "平均查询时间: " << avg_query_time << " 秒\n";
        std::cout << "平均删除和添加总时间: " << avg_sum_delete_add_time << " 秒\n";
        std::cout << "------------------------------------------------------------------" << std::endl;

        // 将结果写入 CSV
        std::vector<std::vector<std::string>> result_data = {{
                                                                     std::to_string(iteration + 1),
                                                                     "N/A",  // 如果可以计算不可达点的数量，调整此值
                                                                     std::to_string(recall),
                                                                     std::to_string(avg_delete_time),
                                                                     std::to_string(avg_add_time),
                                                                     std::to_string(avg_sum_delete_add_time),
                                                                     std::to_string(avg_query_time)
                                                             }};
        util::writeCSVApp(output_csv_path, result_data);
    }

    // 保存更新后的索引
    index.saveIndex(output_index_path);
    return 0;
}
