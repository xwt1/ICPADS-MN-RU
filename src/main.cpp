//
// Created by xiaowentao on 2024/2/14.
//
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <bits/stdc++.h>

#include "hnswlib/hnswlib.h"
#include "file.h"

using namespace mlpack;


int main(){
//    std::vector<Point> points = {
//            {1.0, 2.0},
//            {1.0, 3.0},
//            {10.0, 10.0},
//            {10.0, 11.0} // 新添加的点
//            // 可以继续添加更多点
//    };
//    Graph::BaseGraph g(points,1);
////    g.print();
//    std::cout << "MLPack version: " << mlpack::util::GetVersion() << std::endl;
//
//    std::cout<<1<<std::endl;
//
//    std::cout<<123<<std::endl;
//
//    std::unordered_map<int,int> um;
//    um[1] =2;
//    std::cout<<5435<<std::endl;
//    return 0;

    int dim = 2;               // Dimension of the elements
    int max_elements = 5000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    int query_index = 300;

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(36);
    std::uniform_real_distribution<> distrib_real;
    // 连续存储数据点
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }
    // 建图
    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    using namespace std::chrono; // 使用chrono命名空间，简化代码

    auto start = high_resolution_clock::now(); // 开始时间
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data+query_index, 10);
    auto stop = high_resolution_clock::now(); // 结束时间

    auto duration = duration_cast<microseconds>(stop - start); // 计算持续时间

    std::cout << "Time taken by function: "
    << duration.count() << " microseconds" << std::endl;



    std::vector<std::vector<float>> ans;
    while(!result.empty()){
        auto top = result.top();
        std::vector<float> temp;
        for(auto i =0 ;i < dim;i++){
            temp.push_back(data[top.second * dim + i]);
        }
        ans.push_back(temp);
        result.pop();
    }
    auto Points = WriteOpt::ConvertTo2DVector(data,dim,max_elements);
//    auto ans = WriteOpt::ConvertTo2DVector(data,dim,max_elements);
    auto query = WriteOpt::ConvertTo2DVector(data+query_index,dim,1);

#ifdef  PROJECT_ROOT_DIR
    std::cout<<PROJECT_ROOT_DIR<<std::endl;
    std::vector <std::vector<std::string>> coordinate={{"X","Y","Z"}};
//    std::vector<std::string> temp = {"X","Y","Z"};
//    WriteOpt::WriteCSVFile<std::string>(coordinate,PROJECT_ROOT_DIR"/data/Points.csv");
    WriteOpt::WriteCSVFile<float>(Points,PROJECT_ROOT_DIR"/data/Points.csv");
//    WriteOpt::WriteCSVFile<std::string>(coordinate,PROJECT_ROOT_DIR"/data/ans.csv");
    WriteOpt::WriteCSVFile<float>(ans,PROJECT_ROOT_DIR"/data/ans.csv");
//    WriteOpt::WriteCSVFile<std::string>(coordinate,PROJECT_ROOT_DIR"/data/query.csv");
    WriteOpt::WriteCSVFile<float>(query,PROJECT_ROOT_DIR"/data/query.csv");
    std::system("python3 " PROJECT_ROOT_DIR"/data/draw.py " PROJECT_ROOT_DIR);
#endif
//    WriteOpt::WriteCSVFile(Points,"");


    delete[] data;
    delete alg_hnsw;
    return 0;

}


