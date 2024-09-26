#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sys/stat.h>
#include "util.h"

int main() {
    // 参数设置
    std::string data_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/sift_base.fvecs";
    std::string save_dir = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/data/sift/faiss_index/";
    std::string save_path = save_dir + "sift_index.bin";

    int dimension = 0;
    int total_vectors = 0;  // 总向量数
    int nlist = 1000;       // 聚类中心数

    // 调用自定义的 load_fvecs 函数来读取 fvecs 文件
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dimension, total_vectors);

    // 将 2D std::vector 转换为 1D 数组
    std::vector<float> flat_data;
    for (const auto& vec : data) {
        flat_data.insert(flat_data.end(), vec.begin(), vec.end());
    }

    // 初始化量化器 (使用 L2 距离的暴力搜索来初始化量化器)
    faiss::IndexFlatL2 quantizer(dimension);

    // 初始化 IVF-FLAT 索引
    faiss::IndexIVFFlat index(&quantizer, dimension, nlist, faiss::METRIC_L2);

    // 训练索引 (使用全量数据来训练聚类中心)
    std::cout << "开始训练索引..." << std::endl;
    index.train(total_vectors, flat_data.data());  // 使用展平后的数据进行训练
    std::cout << "索引训练完成。" << std::endl;

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // 一次性添加所有向量到索引
    index.add(total_vectors, flat_data.data());  // 这里传入展平后的数据
    // 结束计时并计算总耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "所有向量已添加到索引。" << std::endl;

    // 检查保存目录是否存在，不存在则创建
    struct stat info;
    if (stat(save_dir.c_str(), &info) != 0) {
        std::cout << "保存目录不存在，创建中..." << std::endl;
        system(("mkdir -p " + save_dir).c_str());
    }

    // 保存索引到文件
    faiss::write_index(&index, save_path.c_str());
    std::cout << "索引已保存到 " << save_path << std::endl;

    // 计算耗时
    std::chrono::duration<double> total_time = end_time - start_time;

    // 将总耗时转换为小时、分钟、秒
    int hours = static_cast<int>(total_time.count()) / 3600;
    int minutes = (static_cast<int>(total_time.count()) % 3600) / 60;
    double seconds = total_time.count() - hours * 3600 - minutes * 60;

    std::cout << "构建索引总耗时：" << hours << "小时 " << minutes << "分钟 " << seconds << "秒" << std::endl;

    return 0;
}
