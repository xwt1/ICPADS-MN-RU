//
// Created by xiaowentao on 2024/2/14.
//
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
//#include
#include <bits/stdc++.h>

#include "base_graph.h"
using namespace mlpack;


int main(){
//    arma::mat data = {
//            {1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5},
//            {2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 4.5}
//    };
//
//    // 初始化DBSCAN算法，设置邻域半径（epsilon）和最小点数（minPoints）
//    mlpack::dbscan::DBSCAN<> dbscan(1.0, 3);
//
//    // 用于存储每个点的聚类结果
//    arma::Row<size_t> assignments;
//
//    // 执行DBSCAN聚类
//    dbscan.Cluster(data, assignments);
//
//    // 打印聚类结果
//    std::cout << "Cluster assignments:" << std::endl;
//    assignments.t().print();
//
//    return 0;



    std::vector<Point> points = {
            {1.0, 2.0},
            {1.0, 3.0},
            {10.0, 10.0},
            {10.0, 11.0} // 新添加的点
            // 可以继续添加更多点
    };
    Graph::BaseGraph g(points);
//    g.print();
    std::cout << "MLPack version: " << mlpack::util::GetVersion() << std::endl;
    return 0;

}