//
// Created by xiaowentao on 2024/2/14.
//
#include <filesystem>

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"
#include "file.h"
#include "evaluate/evaluate.h"

using namespace mlpack;

//void generateIndex(int dim,
//                   int max_elements,
//                   std::string data_file_path,
//                   int M,
//                   int ef_construction,
//                   std::string index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/graph-search/data/hnswindex1.bin"){
//    hnswlib::L2Space space(dim);
//    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
//    float* data = new float[dim * max_elements];
//    std::ifstream file(data_file_path, std::ios::binary);
//    file.read(reinterpret_cast<char*>(data), dim * max_elements * sizeof(float));
//    std::cout<<"开始生成索引"<<std::endl;
//    for (int i = 0; i < max_elements; i++) {
//        alg_hnsw->addPoint(data + i * dim, i);
//    }
//    alg_hnsw->saveIndex(index_file);
//    std::cout<<"索引生成完成"<<std::endl;
//    delete[] data;
//    delete alg_hnsw;
//}

void query_mutiple_time(int k,int dim,int max_elements, std::string data_file_path,int start_index,int times){
    const std::string index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/graph-search/data/hnswindex.bin";
    hnswlib::L2Space space(dim);
    auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file,false);

    float* data = new float[dim * max_elements];
//    float* data = new float[dim * 10];
    std::ifstream file(data_file_path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data), dim * max_elements * sizeof(float));

    float mmr_score = 0;
    for(int i = start_index; i < start_index+times; i++){
        using namespace std::chrono; // 使用chrono命名空间，简化代码

//        auto start = high_resolution_clock::now(); // 开始时间
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+i*dim, k);
//        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->D_searchKnn(data+i*dim, k,hnswlib::diversity_type::MMR,1);
//        auto stop = high_resolution_clock::now(); // 结束时间

//        auto duration = duration_cast<microseconds>(stop - start); // 计算持续时间

        std::vector<std::vector<float>> ans;
        std::vector<float> q_point;
        for(auto j =0 ;j < dim;j++){
            q_point.push_back(data[i * dim + j]);
        }
        while(!result.empty()){
            auto top = result.top();
            std::vector<float> temp;
            for(auto j =0 ;j < dim;j++){
                temp.push_back(data[top.second * dim + j]);
            }
            ans.push_back(temp);
            result.pop();
        }
        std::cout<<ans.size()<<std::endl;
        for(auto i:ans){
            for(auto j:i){
                std::cout<<j<<" ";
            }
            std::cout<<std::endl;
        }
        mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8,Evaluate::distance);
    }



//    std::cout << "Time taken by function: "
//              << duration.count() << " microseconds" << std::endl;

    std::cout<<"mmr_score is: "<<mmr_score / (times)<<std::endl;
}

//void query_myself(int dim,int max_elements, std::string data_file_path,int query_index){
//    const std::string index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/graph-search/data/hnswindex.bin";
//    hnswlib::L2Space space(dim);
//    auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file,false);
//
//    float* data = new float[dim * max_elements];
////    float* data = new float[dim * 10];
//    std::ifstream file(data_file_path, std::ios::binary);
//    file.read(reinterpret_cast<char*>(data), dim * max_elements * sizeof(float));
//
//    using namespace std::chrono; // 使用chrono命名空间，简化代码
//
//    auto start = high_resolution_clock::now(); // 开始时间
////    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+10, 100);
//    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->D_searchKnn(data+10, 100,hnswlib::diversity_type::MMR,1);
//    auto stop = high_resolution_clock::now(); // 结束时间
//
//    auto duration = duration_cast<microseconds>(stop - start); // 计算持续时间
//
//    std::vector<std::vector<float>> ans;
//    std::vector<float> q_point;
//    for(auto i =0 ;i < dim;i++){
//        q_point.push_back(data[query_index * dim + i]);
//    }
//    while(!result.empty()){
//        auto top = result.top();
//        std::vector<float> temp;
//        for(auto i =0 ;i < dim;i++){
//            temp.push_back(data[top.second * dim + i]);
//
//        }
//        ans.push_back(temp);
//        result.pop();
//    }
//    auto mmr_score = Evaluate::evaluateWithMMR<float>(ans,q_point,0.8);
//
//    std::cout << "Time taken by function: "
//    << duration.count() << " microseconds" << std::endl;
//
//    std::cout<<"mmr_score is: "<<mmr_score<<std::endl;
//}

//生成索引
//int main(int argc, char* argv[]){
//    // 检查参数数量
//    if (argc != 4) {
//        std::cerr << "Usage: " << argv[0] << " <dimension> <num_points> <file_path>" << std::endl;
//        return -1;
//    }
//
//    int dim = std::stoi(argv[1]);
//    int max_elements = std::stoi(argv[2]);
//    std::string data_file_path = argv[3];
//
//    // 检查文件路径是否存在
//    if (!std::filesystem::exists(data_file_path)) {
//        std::cerr << "File does not exist: " << data_file_path << std::endl;
//        return 1;
//    }
//
//    // 检查这是否确实是一个文件而不是目录
//    if (!std::filesystem::is_regular_file(data_file_path)) {
//        std::cerr << "Path is not a file: " << data_file_path << std::endl;
//        return 1;
//    }
//    generateIndex(dim,max_elements,data_file_path,16,200);
//}


//执行查询
int main(int argc, char* argv[]){
    // 检查参数数量
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dimension> <num_points> <file_path>" << std::endl;
        return -1;
    }
    int dim = std::stoi(argv[1]);
    int max_elements = std::stoi(argv[2]);
    std::string data_file_path = argv[3];
    // 检查文件路径是否存在
    if (!std::filesystem::exists(data_file_path)) {
        std::cerr << "File does not exist: " << data_file_path << std::endl;
        return 1;
    }

    // 检查这是否确实是一个文件而不是目录
    if (!std::filesystem::is_regular_file(data_file_path)) {
        std::cerr << "Path is not a file: " << data_file_path << std::endl;
        return 1;
    }
//    query_myself(dim,max_elements,data_file_path,2);
    query_mutiple_time(2,dim,max_elements,data_file_path,114514,1);
}


//int main(int argc, char* argv[]){
//
////    std::vector<Point> points = {
////            {1.0, 2.0},
////            {1.0, 3.0},
////            {10.0, 10.0},
////            {10.0, 11.0} // 新添加的点
////            // 可以继续添加更多点
////    };
////    Graph::BaseGraph g(points,1);
//////    g.print();
////    std::cout << "MLPack version: " << mlpack::util::GetVersion() << std::endl;
////
////    std::cout<<1<<std::endl;
////
////    std::cout<<123<<std::endl;
////
////    std::unordered_map<int,int> um;
////    um[1] =2;
////    std::cout<<5435<<std::endl;
////    return 0;
//    // 检查参数数量
//    if (argc != 4) {
//        std::cerr << "Usage: " << argv[0] << " <dimension> <num_points> <file_path>" << std::endl;
//        return -1;
//    }
//
//    int dim = std::stoi(argv[1]);
//    int max_elements = std::stoi(argv[2]);
//    std::string data_file_path = argv[3];
//
//    // 检查文件路径是否存在
//    if (!std::filesystem::exists(data_file_path)) {
//        std::cerr << "File does not exist: " << data_file_path << std::endl;
//        return 1;
//    }
//
//    // 检查这是否确实是一个文件而不是目录
//    if (!std::filesystem::is_regular_file(data_file_path)) {
//        std::cerr << "Path is not a file: " << data_file_path << std::endl;
//        return 1;
//    }
//
//
////    int dim = 2;               // Dimension of the elements
////    int max_elements = 5000;   // Maximum number of elements, should be known beforehand
//    int M = 16;                 // Tightly connected with internal dimensionality of the data
//    // strongly affects the memory consumption
//    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
//
//    int query_index = 300;
//
//    // Initing index
//    const std::string index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/graph-search/data/hnswindex.bin";
//    hnswlib::L2Space space(dim);
////    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
//
////    // Generate random data
////    std::mt19937 rng;
////    rng.seed(36);
////    std::uniform_real_distribution<> distrib_real;
////    // 连续存储数据点
////    float* data = new float[dim * max_elements];
////    for (int i = 0; i < dim * max_elements; i++) {
////        data[i] = distrib_real(rng);
////    }
//
//    auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file,false);
//
//    using namespace std::chrono; // 使用chrono命名空间，简化代码
//
//    auto start1 = high_resolution_clock::now(); // 开始时间
//
//    float* data = new float[dim * max_elements];
//    std::ifstream file(data_file_path, std::ios::binary);
//    file.read(reinterpret_cast<char*>(data), dim * max_elements * sizeof(float));
//    std::cout<<*(data+1)<<std::endl;
//    float pp1 = 1.2;
//    std::cout<<pp1<<std::endl;
//
//    std::cout<<"读取完毕"<<'\n';
//    auto stop1 = high_resolution_clock::now(); // 结束时间
//
//    auto duration1 = duration_cast<microseconds>(stop1 - start1); // 计算持续时间
//    std::cout << "Time taken by function: "
//              << duration1.count() << " microseconds" << std::endl;
//
//    // 建图
//    // Add data to index
//
////    for (int i = 0; i < max_elements; i++) {
////        alg_hnsw->addPoint(data + i * dim, i);
////    }
//
//
//    alg_hnsw->saveIndex(index_file);
//
//    using namespace std::chrono; // 使用chrono命名空间，简化代码
//
//    auto start = high_resolution_clock::now(); // 开始时间
//    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+query_index, 100);
//    auto stop = high_resolution_clock::now(); // 结束时间
//
//    auto duration = duration_cast<microseconds>(stop - start); // 计算持续时间
//
//    std::cout << "Time taken by function: "
//    << duration.count() << " microseconds" << std::endl;
//
//
//
//    std::vector<std::vector<float>> ans;
//    while(!result.empty()){
//        auto top = result.top();
//        std::vector<float> temp;
//        for(auto i =0 ;i < dim;i++){
//            temp.push_back(data[top.second * dim + i]);
//            std::cout<<top.second<<",";
//            std::cout<<temp[i]<<" ";
//        }
//        std::cout<<std::endl;
//        ans.push_back(temp);
//        result.pop();
//    }
//    auto Points = WriteOpt::ConvertTo2DVector(data,dim,max_elements);
////    auto ans = WriteOpt::ConvertTo2DVector(data,dim,max_elements);
//    auto query = WriteOpt::ConvertTo2DVector(data+query_index,dim,1);
//
//#ifdef  PROJECT_ROOT_DIR
//    std::cout<<PROJECT_ROOT_DIR<<std::endl;
//    std::vector <std::vector<std::string>> coordinate={{"X","Y","Z"}};
////    std::vector<std::string> temp = {"X","Y","Z"};
////    WriteOpt::WriteCSVFile<std::string>(coordinate,PROJECT_ROOT_DIR"/data/Points.csv");
//    WriteOpt::WriteCSVFile<float>(Points,PROJECT_ROOT_DIR"/data/Points.csv");
////    WriteOpt::WriteCSVFile<std::string>(coordinate,PROJECT_ROOT_DIR"/data/ans.csv");
//    WriteOpt::WriteCSVFile<float>(ans,PROJECT_ROOT_DIR"/data/ans.csv");
////    WriteOpt::WriteCSVFile<std::string>(coordinate,PROJECT_ROOT_DIR"/data/query.csv");
//    WriteOpt::WriteCSVFile<float>(query,PROJECT_ROOT_DIR"/data/query.csv");
////    std::system("python3 " PROJECT_ROOT_DIR"/data/draw.py " PROJECT_ROOT_DIR);
//#endif
////    WriteOpt::WriteCSVFile(Points,"");
//
//
//    delete[] data;
//    delete alg_hnsw;
//    return 0;
//
//}


