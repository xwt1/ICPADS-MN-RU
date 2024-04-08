//
// Created by root on 3/27/24.
//
#include <filesystem>

#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"
#include "file.h"
#include "evaluate/evaluate.h"

void query_sift_ip(int k,int dim,int base_max_elements,int query_max_elements, std::string query_vector_file,std::string base_vector_file,std::string index_file){
    hnswlib::InnerProductSpace InnerProductSpace(dim);
    auto alg_hnsw = new hnswlib::DHierarchicalNSW<float>(&InnerProductSpace, index_file,hnswlib::ip,hnswlib::ip,false);

    int base_total_vectors = -1;
    float* base_data = new float[dim*base_max_elements];
    if(!ReadOpt::ReadFvecsFileIntoArray<float>(base_vector_file,base_data,base_total_vectors,dim,base_max_elements)){
        return;
    }
    float* query_data = new float[dim * query_max_elements];
    int totalVectors = -1;
    if(ReadOpt::ReadFvecsFileIntoArray<float>(query_vector_file,query_data,totalVectors,dim,query_max_elements)){
        using namespace std::chrono;
        float raw_mmr_score = 0;
        float Dhnsw_mmr_score = 0;
        float raw_mmr_ILAD_score = 0;
        float Dhnsw_mmr_ILAD_score = 0;
        float raw_mmr_ILMD_score = 0;
        float Dhnsw_mmr_ILMD_score = 0;
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
//            std::cout<<ans.size()<<std::endl;
//            for(auto i:ans){
//                for(auto j:i){
//                    std::cout<<j<<" ";
//                }
//                std::cout<<std::endl;
//            }
            Dhnsw_mmr_score += Evaluate::evaluateWithMMR<float>(ans,q_point,0.8,hnswlib::ip,hnswlib::ip);
            Dhnsw_mmr_ILAD_score += Evaluate::evaluateWithILAD<float>(ans,hnswlib::distance);
            Dhnsw_mmr_ILMD_score += Evaluate::evaluateWithILMD<float>(ans,hnswlib::distance);
        }
        for(int i = 0 ; i < totalVectors; i++){
            //        计算raw
            auto start = high_resolution_clock::now(); // 开始时间
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query_data+i*dim, k);
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
            auto result_ans = Evaluate::obtainRawMMR<float>(result1,q_point,0.8,k,hnswlib::ip,hnswlib::ip);
            stop = high_resolution_clock::now(); // 结束时间

//            std::cout<<result_ans.size()<<std::endl;
//            for(auto i:result_ans){
//                for(auto j:i){
//                    std::cout<<j<<" ";
//                }
//                std::cout<<std::endl;
//            }

            raw_mmr_duration += duration_cast<microseconds>(stop - start); // 计算持续时间
            raw_mmr_score += Evaluate::evaluateWithMMR<float>(result_ans,q_point,0.8,hnswlib::ip,hnswlib::ip);
            raw_mmr_ILAD_score += Evaluate::evaluateWithILAD<float>(result_ans,hnswlib::distance);
            raw_mmr_ILMD_score += Evaluate::evaluateWithILMD<float>(result_ans,hnswlib::distance);
        }
        std::cout << "Time taken by raw: "
                  << raw_mmr_duration.count() << " microseconds" << std::endl;

        std::cout << "Time taken by Dhnsw: "
                  << Dhnsw_mmr_duration.count() << " microseconds" << std::endl;

        std::cout<<"raw_mmr_score is: "<<raw_mmr_score / (float)(totalVectors)<<std::endl;
        std::cout<<"raw_mmr_ILAD_score is: "<<raw_mmr_ILAD_score / (float)(totalVectors)<<std::endl;
        std::cout<<"raw_mmr_ILMD_score is: "<<raw_mmr_ILMD_score / (float)(totalVectors)<<std::endl;

        std::cout<<"Dhnsw_mmr_score is: "<<Dhnsw_mmr_score / (float)(totalVectors)<<std::endl;
        std::cout<<"Dhnsw_mmr_ILAD_score is: "<<Dhnsw_mmr_ILAD_score / (float)(totalVectors)<<std::endl;
        std::cout<<"Dhnsw_mmr_ILMD_score is: "<<Dhnsw_mmr_ILMD_score / (float)(totalVectors)<<std::endl;
    }
    delete[] base_data;
    delete[] query_data;
    delete alg_hnsw;
}


int main(int argc, char* argv[]) {
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

    query_sift_ip(100,dim,base_max_elements,query_max_elements,query_vector_file,base_vector_file,index_file);
    return 0;
}
