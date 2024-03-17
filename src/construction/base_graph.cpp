//
// Created by xiaowentao on 2024/2/15.
//

#include "base_graph.h"

//#include "point.h"

Graph::BaseGraph::BaseGraph(std::vector<Point> data, size_t cluster_category_id){


    switch (cluster_category_id) {
        case 0:{
            arma::mat mat_data = util::Util::ConvertPointsToArmaMat(data);
            mlpack::dbscan::DBSCAN<> dbscan(4.0, 4); // 示例参数：半径为1.0，最小点数为3
            arma::Row<size_t> assignments; // 用来存储每个点的簇分配结果
            dbscan.Cluster(mat_data, assignments);
            // 输出簇分配结果
            assignments.t().print();
        }
        break;
        case 1:{


            // 创建 KMeans 对象
            //    mlpack::kmeans::KMeans<> kmeans;
            mlpack::kmeans::KMeans<> kmeans;

            arma::mat mat_data = util::Util::ConvertPointsToArmaMat(data);

            size_t clusters = std::min(data.size(),(unsigned long)2);

            // 运行 K-Means 算法
            arma::Row<size_t> assignments; // 用于存储每个数据点的聚类分配
            arma::mat centroids; // 用于存储聚类中心
            kmeans.Cluster(mat_data, clusters, assignments, centroids);

            // 输出结果
            centroids.print("Centroids:");
            assignments.print("Assignments:");
        }
        break;
    }




}

Graph::BaseGraph::BaseGraph() {

}

unsigned short Graph::BaseGraphCluster::GetClusterId() const {
    return cluster_id_;
}

Graph::BaseGraphCluster::BaseGraphCluster(){

}

Graph::BaseGraphCluster::BaseGraphCluster(unsigned short cluster_id , std::vector <std::shared_ptr<Node> >& node):cluster_id_(cluster_id){
//    if
}

void Graph::BaseGraphCluster::GetFurthestDis(std::vector <std::shared_ptr<Node> >& node){
    
}

