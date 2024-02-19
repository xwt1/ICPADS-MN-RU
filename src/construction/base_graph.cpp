//
// Created by xiaowentao on 2024/2/15.
//

#include <base_graph.h>

Graph::BaseGraph::BaseGraph(std::vector<Point> data){
    arma::mat mat_data = Util::ConvertPointsToArmaMat(data);
    mlpack::dbscan::DBSCAN<> dbscan(4.0, 2); // 示例参数：半径为1.0，最小点数为3
    arma::Row<size_t> assignments; // 用来存储每个点的簇分配结果
    dbscan.Cluster(mat_data, assignments);
    // 输出簇分配结果
//    assignments.print("Cluster assignments:");
    assignments.t().print();

}

Graph::BaseGraph::BaseGraph() {

}

