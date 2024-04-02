//
// Created by root on 3/21/24.
//
#include <filesystem>

#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"
#include "file.h"
#include "evaluate/evaluate.h"

void query_mutiple_time(int k,int dim,int max_elements, std::string data_file_path,int start_index,int times){
    const std::string index_file = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/graph-search/data/hnswindex.bin";
    hnswlib::L2Space space(dim);
    auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file,false);
    float* data = new float[dim * max_elements];
    std::ifstream file(data_file_path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data), dim * max_elements * sizeof(float));
    using namespace std::chrono; // 使用chrono命名空间，简化代码
    float raw_mmr_score = 0;
    float Dhnsw_mmr_score = 0;
    auto raw_mmr_duration = std::chrono::microseconds(0);
    auto Dhnsw_mmr_duration = std::chrono::microseconds(0);
    for(int i = start_index; i < start_index+times; i++){
//        计算Dhnsw
        auto start = high_resolution_clock::now(); // 开始时间
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->D_searchKnn(data+i*dim, k,hnswlib::diversity_type::MMR,1);
        auto stop = high_resolution_clock::now(); // 结束时间
        Dhnsw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
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
//        std::cout<<ans.size()<<std::endl;
//        for(auto i:ans){
//            for(auto j:i){
//                std::cout<<j<<" ";
//            }
//            std::cout<<std::endl;
//        }
        Dhnsw_mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8,Evaluate::distance);
    }
    for(int i = start_index; i < start_index+times; i++){
//        计算raw
        auto start = high_resolution_clock::now(); // 开始时间
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+i*dim, k);
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
        std::cout<<"result1 size: "<<result1.size()<<std::endl;
        for(auto i:result1){
            for(auto j:i){
                std::cout<<j<<" ";
            }
            std::cout<<std::endl;
        }


        std::vector<float> q_point;
        for(auto j =0 ;j < dim;j++){
            q_point.push_back(data[i * dim + j]);
        }
        start = high_resolution_clock::now(); // 开始时间
        auto result_ans = Evaluate::obtainRawMMR<float>(result1,q_point,0.8,k);
        stop = high_resolution_clock::now(); // 结束时间

        std::cout<<result_ans.size()<<std::endl;
        for(auto i:result_ans){
            for(auto j:i){
                std::cout<<j<<" ";
            }
            std::cout<<std::endl;
        }

        raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
        raw_mmr_score += Evaluate::evaluateWithMMR<float>(result_ans,q_point,0.8,Evaluate::distance);
    }


    std::cout << "Time taken by raw: "
              << raw_mmr_duration.count() << " microseconds" << std::endl;

    std::cout << "Time taken by Dhnsw: "
              << Dhnsw_mmr_duration.count() << " microseconds" << std::endl;

    std::cout<<"raw_mmr_score is: "<<raw_mmr_score / (times)<<std::endl;

    std::cout<<"Dhnsw_mmr_score is: "<<Dhnsw_mmr_score / (times)<<std::endl;
}

void query_sift(int k,int dim,int base_max_elements,int query_max_elements, std::string query_vector_file,std::string base_vector_file,std::string index_file){
    hnswlib::L2Space space(dim);
    auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_file,false);

    int base_total_vectors = -1;
    float* base_data = new float[dim*base_max_elements];
    if(!ReadOpt::ReadFvecsFileIntoArray<float>(base_vector_file,base_data,base_total_vectors,dim,base_max_elements)){
        return;
    }
    float* query_data = new float[dim * query_max_elements];
    int totalVectors = -1;
    if(ReadOpt::ReadFvecsFileIntoArray<float>(query_vector_file,query_data,totalVectors,dim,query_max_elements)){
        using namespace std::chrono; // 使用chrono命名空间，简化代码
        float raw_mmr_score = 0;
        float Dhnsw_mmr_score = 0;
        auto raw_mmr_duration = std::chrono::microseconds(0);
        auto Dhnsw_mmr_duration = std::chrono::microseconds(0);
        for(int i = 0 ; i < totalVectors; i++){
//            for(int j = 0; j< dim ;j++){
//                std::cout<<*(query_data+i*dim +j)<<" ";
//            }
//            std::cout<<std::endl;
            //        计算Dhnsw
            auto start = high_resolution_clock::now(); // 开始时间
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->D_searchKnn(query_data+i*dim, k,hnswlib::diversity_type::MMR,1);
            auto stop = high_resolution_clock::now(); // 结束时间
            Dhnsw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
            std::vector<std::vector<float>> ans;
            std::vector<float> q_point;
            for(auto j =0 ;j < dim;j++){
                q_point.push_back(query_data[i * dim + j]);
            }
            while(!result.empty()){
                auto top = result.top();
                std::vector<float> temp;
                for(auto j =0 ;j < dim;j++){
                    temp.push_back(base_data[top.second * dim + j]);
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
            Dhnsw_mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8,Evaluate::distance);
        }
        for(int i = 0 ; i < totalVectors; i++){
            //        计算raw
            auto start = high_resolution_clock::now(); // 开始时间
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query_data+i*dim, k*10);
            auto stop = high_resolution_clock::now(); // 结束时间
            raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
            std::vector <std::vector <float>> result1;
            while(!result.empty()){
                auto top = result.top();
                std::vector<float> temp;
                for(auto j =0 ;j < dim;j++){
                    temp.push_back(base_data[top.second * dim + j]);
                }
                result1.push_back(temp);
                result.pop();
            }
//            std::cout<<"result1 size: "<<result1.size()<<std::endl;
//            for(auto i:result1){
//                for(auto j:i){
//                    std::cout<<j<<" ";
//                }
//                std::cout<<std::endl;
//            }


            std::vector<float> q_point;
            for(auto j =0 ;j < dim;j++){
                q_point.push_back(query_data[i * dim + j]);
            }
            start = high_resolution_clock::now(); // 开始时间
            auto result_ans = Evaluate::obtainRawMMR<float>(result1,q_point,0.8,k);
            stop = high_resolution_clock::now(); // 结束时间

//            std::cout<<result_ans.size()<<std::endl;
//            for(auto i:result_ans){
//                for(auto j:i){
//                    std::cout<<j<<" ";
//                }
//                std::cout<<std::endl;
//            }

            raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
            raw_mmr_score += Evaluate::evaluateWithMMR<float>(result_ans,q_point,0.8,Evaluate::distance);
        }
        std::cout << "Time taken by raw: "
                  << raw_mmr_duration.count() << " microseconds" << std::endl;

        std::cout << "Time taken by Dhnsw: "
                  << Dhnsw_mmr_duration.count() << " microseconds" << std::endl;

        std::cout<<"raw_mmr_score is: "<<raw_mmr_score / (float)(totalVectors)<<std::endl;

        std::cout<<"Dhnsw_mmr_score is: "<<Dhnsw_mmr_score / (float)(totalVectors)<<std::endl;
    }
    delete[] base_data;
    delete[] query_data;
    delete alg_hnsw;
}

int main(int argc, char* argv[]){
    // 检查参数数量
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <dimension> <base_max_elements> <query_max_elements> <query_vector_file> <base_vector_file> <index_file>" << std::endl;
        return -1;
    }
    int dim = std::stoi(argv[1]);
    int base_max_elements = std::stoi(argv[2]);
    int query_max_elements = std::stoi(argv[3]);
    std::string query_vector_file = argv[4];
    std::string base_vector_file = argv[5];
    std::string index_file = argv[6];
    // 检查文件路径是否存在
    if (!std::filesystem::exists(query_vector_file)) {
        std::cerr << "File does not exist: " << query_vector_file << std::endl;
        return 1;
    }

    // 检查这是否确实是一个文件而不是目录
    if (!std::filesystem::is_regular_file(query_vector_file)) {
        std::cerr << "Path is not a file: " << query_vector_file << std::endl;
        return 1;
    }
    // 检查文件路径是否存在
    if (!std::filesystem::exists(index_file)) {
        std::cerr << "File does not exist: " << index_file << std::endl;
        return 1;
    }

    // 检查这是否确实是一个文件而不是目录
    if (!std::filesystem::is_regular_file(index_file)) {
        std::cerr << "Path is not a file: " << index_file << std::endl;
        return 1;
    }

    query_sift(100,dim,base_max_elements,query_max_elements,query_vector_file,base_vector_file,index_file);
//    query_mutiple_time(2,dim,max_elements,data_file_path,114514,1);
    return 0;
}

//int main(){
//    int dim = 2;               // Dimension of the elements
//    int max_elements = 1000;   // Maximum number of elements, should be known beforehand
//    int M = 16;                 // Tightly connected with internal dimensionality of the data
//    // strongly affects the memory consumption
//    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
//
//    // Initing index
//    hnswlib::L2Space space(dim);
//    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
//
//    // Generate random data
//    std::mt19937 rng;
//    rng.seed(47);
//    std::uniform_real_distribution<> distrib_real;
//    float* data = new float[dim * max_elements];
//    for (int i = 0; i < dim * max_elements; i++) {
//        data[i] = distrib_real(rng);
//    }
//
//    // Add data to index
//    for (int i = 0; i < max_elements; i++) {
//        alg_hnsw->addPoint(data + i * dim, i);
//    }
//    using namespace std::chrono; // 使用chrono命名空间，简化代码
//    int k=max_elements/100;
//    float raw_mmr_score = 0;
//    float Dhnsw_mmr_score = 0;
//    float hnsw_mmr_score = 0;
//    auto raw_mmr_duration = std::chrono::microseconds(0);
//    auto Dhnsw_mmr_duration = std::chrono::microseconds(0);
//    int index = 10;
//    {
////        hnsw
//        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+index*dim, k);
//        std::vector<std::vector<float>> ans;
//        std::vector<float> q_point;
//        for(auto j =0 ;j < dim;j++){
//            q_point.push_back(data[index * dim + j]);
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
//        hnsw_mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8);
//    }
//    {
////        raw
//        auto start = high_resolution_clock::now(); // 开始时间
//        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+index*dim, k*10);
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
////        std::cout<<"result1 size: "<<result1.size()<<std::endl;
////        for(auto i:result1){
////            for(auto j:i){
////                std::cout<<j<<" ";
////            }
////            std::cout<<std::endl;
////        }
//
//
//        std::vector<float> q_point;
//        for(auto j =0 ;j < dim;j++){
//            q_point.push_back(data[index * dim + j]);
//        }
//        start = high_resolution_clock::now(); // 开始时间
//        auto result_ans = Evaluate::obtainRawMMR<float>(result1,q_point,0.8,k);
//        stop = high_resolution_clock::now(); // 结束时间
//
//        std::cout<<result_ans.size()<<std::endl;
////        for(auto i:result_ans){
////            for(auto j:i){
////                std::cout<<j<<" ";
////            }
////            std::cout<<std::endl;
////        }
//
//        raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
//        raw_mmr_score += Evaluate::evaluateWithMMR<float>(result_ans,q_point,0.8);
//    }
////    int count = 0;
////    std::vector <bool> vis(max_elements,false);
////    alg_hnsw->dfs(1,vis,count);
////    std::cout<<"count:"<<count<<std::endl;
//    {
//
//        auto start = high_resolution_clock::now(); // 开始时间
//        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->D_searchKnn(data+index*dim, k,hnswlib::diversity_type::MMR,1);
//        auto stop = high_resolution_clock::now(); // 结束时间
//        Dhnsw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
//        std::vector<std::vector<float>> ans;
//        std::vector<float> q_point;
//        for(auto j =0 ;j < dim;j++){
//            q_point.push_back(data[index * dim + j]);
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
//    std::cout << "Time taken by raw: "
//          << raw_mmr_duration.count() << " microseconds" << std::endl;
//
//    std::cout << "Time taken by Dhnsw: "
//              << Dhnsw_mmr_duration.count() << " microseconds" << std::endl;
//
//    std::cout<<"raw_mmr_score is: "<<raw_mmr_score / (1)<<std::endl;
//
//    std::cout<<"Dhnsw_mmr_score is: "<<Dhnsw_mmr_score / (1)<<std::endl;
//
//
//    std::cout<<"hnsw_mmr_score is: "<<hnsw_mmr_score / (1)<<std::endl;
//}