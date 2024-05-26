//
// Created by root on 5/16/24.
//
#include "hnswlib/hnswlib.h"
#include "bits/stdc++.h"
#include <iostream>
#include <string>


int main() {
    try {
        // 创建空间接口（假设是L2距离）
        hnswlib::L2Space l2space(2); // 2维数据
        size_t max_elements = 1000000;

        // 创建HNSW索引
        hnswlib::HierarchicalNSW<float> hnsw(&l2space, max_elements);

        // 从文件加载索引
        std::string index_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_index/delete_update.bin";
        hnsw.loadIndex(index_path, &l2space);

        // 导出图的层次结构
        std::string graph_export_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/graph.txt";
        hnsw.exportGraphNodes(graph_export_path);

        std::cout << "Graph nodes have been exported to " << graph_export_path << std::endl;
    } catch (const std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
