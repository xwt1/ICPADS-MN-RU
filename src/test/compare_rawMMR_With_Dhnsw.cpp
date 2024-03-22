//
// Created by root on 3/21/24.
//
#include <filesystem>

#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"
#include "file.h"
#include "evaluate/evaluate.h"

//void query_mutiple_time(int k,int dim,int max_elements, std::string data_file_path,int start_index,int times){
//    const std::string index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/graph-search/data/hnswindex.bin";
//    hnswlib::L2Space space(dim);
//    auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file,false);
//
//    float* data = new float[dim * max_elements];
//    std::ifstream file(data_file_path, std::ios::binary);
//    file.read(reinterpret_cast<char*>(data), dim * max_elements * sizeof(float));
//    using namespace std::chrono; // 使用chrono命名空间，简化代码
//    float raw_mmr_score = 0;
//    float Dhnsw_mmr_score = 0;
//    auto raw_mmr_duration = std::chrono::microseconds(0);
//    auto Dhnsw_mmr_duration = std::chrono::microseconds(0);
//    for(int i = start_index; i < start_index+times; i++){
////        计算Dhnsw
//        auto start = high_resolution_clock::now(); // 开始时间
//        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->D_searchKnn(data+i*dim, k,hnswlib::diversity_type::MMR,1);
//        auto stop = high_resolution_clock::now(); // 结束时间
//        Dhnsw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
//        std::vector<std::vector<float>> ans;
//        std::vector<float> q_point;
//        for(auto j =0 ;j < dim;j++){
//            q_point.push_back(data[i * dim + j]);
//        }
//        while(!result.empty()){
//            auto top = result.top();
//            std::vector<float> temp;
//            for(auto j =0 ;j < dim;j++){
//                temp.push_back(data[top.second * dim + j]);
//            }
//            ans.push_back(temp);
//            result.pop();
//        }
////        std::cout<<ans.size()<<std::endl;
////        for(auto i:ans){
////            for(auto j:i){
////                std::cout<<j<<" ";
////            }
////            std::cout<<std::endl;
////        }
//        Dhnsw_mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8);
//    }
//    for(int i = start_index; i < start_index+times; i++){
////        计算raw
//        auto start = high_resolution_clock::now(); // 开始时间
//        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+i*dim, k);
//        auto stop = high_resolution_clock::now(); // 结束时间
//        raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
//        std::vector <std::vector <float>> result1;
//        while(!result.empty()){
//            auto top = result.top();
//            std::vector<float> temp;
//            for(auto j =0 ;j < dim;j++){
//                temp.push_back(data[top.second * dim + j]);
//            }
//            result1.push_back(temp);
//            result.pop();
//        }
//        std::cout<<"result1 size: "<<result1.size()<<std::endl;
//        for(auto i:result1){
//            for(auto j:i){
//                std::cout<<j<<" ";
//            }
//            std::cout<<std::endl;
//        }
//
//
//        std::vector<float> q_point;
//        for(auto j =0 ;j < dim;j++){
//            q_point.push_back(data[i * dim + j]);
//        }
//        start = high_resolution_clock::now(); // 开始时间
//        auto result_ans = Evaluate::obtainRawMMR<float>(result1,q_point,0.8,k);
//        stop = high_resolution_clock::now(); // 结束时间
//
//        std::cout<<result_ans.size()<<std::endl;
//        for(auto i:result_ans){
//            for(auto j:i){
//                std::cout<<j<<" ";
//            }
//            std::cout<<std::endl;
//        }
//
//        raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
//        raw_mmr_score += Evaluate::evaluateWithMMR<float>(result_ans,q_point,0.8);
//    }
//
//
//    std::cout << "Time taken by raw: "
//              << raw_mmr_duration.count() << " microseconds" << std::endl;
//
//    std::cout << "Time taken by Dhnsw: "
//              << Dhnsw_mmr_duration.count() << " microseconds" << std::endl;
//
//    std::cout<<"raw_mmr_score is: "<<raw_mmr_score / (times)<<std::endl;
//
//    std::cout<<"Dhnsw_mmr_score is: "<<Dhnsw_mmr_score / (times)<<std::endl;
//}
//
//int main(int argc, char* argv[]){
//    // 检查参数数量
//    if (argc != 4) {
//        std::cerr << "Usage: " << argv[0] << " <dimension> <num_points> <file_path>" << std::endl;
//        return -1;
//    }
//    int dim = std::stoi(argv[1]);
//    int max_elements = std::stoi(argv[2]);
//    std::string data_file_path = argv[3];
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
//    query_mutiple_time(2,dim,max_elements,data_file_path,114514,1);
//    return 0;
//}

int main(){
    int dim = 2;               // Dimension of the elements
    int max_elements = 1000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }
    using namespace std::chrono; // 使用chrono命名空间，简化代码
    int k=max_elements/10;
    float raw_mmr_score = 0;
    float Dhnsw_mmr_score = 0;
    float hnsw_mmr_score = 0;
    auto raw_mmr_duration = std::chrono::microseconds(0);
    auto Dhnsw_mmr_duration = std::chrono::microseconds(0);
    int index = 1;
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+index*dim, k);
        std::vector<std::vector<float>> ans;
        std::vector<float> q_point;
        for(auto j =0 ;j < dim;j++){
            q_point.push_back(data[index * dim + j]);
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
        hnsw_mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8);
    }
    {
//        raw
        auto start = high_resolution_clock::now(); // 开始时间
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+index*dim, k);
        auto stop = high_resolution_clock::now(); // 结束时间
        raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
        std::vector <std::vector <float>> result1;
        while(!result.empty()){
            auto top = result.top();
            std::vector<float> temp;
            for(auto j =0 ;j < dim;j++){
                temp.push_back(data[top.second * dim + j]);
            }
            result1.push_back(temp);
            result.pop();
        }
//        std::cout<<"result1 size: "<<result1.size()<<std::endl;
//        for(auto i:result1){
//            for(auto j:i){
//                std::cout<<j<<" ";
//            }
//            std::cout<<std::endl;
//        }


        std::vector<float> q_point;
        for(auto j =0 ;j < dim;j++){
            q_point.push_back(data[index * dim + j]);
        }
        start = high_resolution_clock::now(); // 开始时间
        auto result_ans = Evaluate::obtainRawMMR<float>(result1,q_point,0.8,k);
        stop = high_resolution_clock::now(); // 结束时间

        std::cout<<result_ans.size()<<std::endl;
//        for(auto i:result_ans){
//            for(auto j:i){
//                std::cout<<j<<" ";
//            }
//            std::cout<<std::endl;
//        }

        raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
        raw_mmr_score += Evaluate::evaluateWithMMR<float>(result_ans,q_point,0.8);
    }
//    int count = 0;
//    std::vector <bool> vis(max_elements,false);
//    alg_hnsw->dfs(1,vis,count);
//    std::cout<<"count:"<<count<<std::endl;
    {

        auto start = high_resolution_clock::now(); // 开始时间
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->D_searchKnn(data+index*dim, k,hnswlib::diversity_type::MMR,1);
        auto stop = high_resolution_clock::now(); // 结束时间
        Dhnsw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
        std::vector<std::vector<float>> ans;
        std::vector<float> q_point;
        for(auto j =0 ;j < dim;j++){
            q_point.push_back(data[index * dim + j]);
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
//        std::cout<<ans.size()<<std::endl;
//        for(auto i:ans){
//            for(auto j:i){
//                std::cout<<j<<" ";
//            }
//            std::cout<<std::endl;
//        }
        Dhnsw_mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8);
    }
    std::cout << "Time taken by raw: "
          << raw_mmr_duration.count() << " microseconds" << std::endl;

    std::cout << "Time taken by Dhnsw: "
              << Dhnsw_mmr_duration.count() << " microseconds" << std::endl;

    std::cout<<"raw_mmr_score is: "<<raw_mmr_score / (1)<<std::endl;

    std::cout<<"Dhnsw_mmr_score is: "<<Dhnsw_mmr_score / (1)<<std::endl;


    std::cout<<"hnsw_mmr_score is: "<<hnsw_mmr_score / (1)<<std::endl;
}