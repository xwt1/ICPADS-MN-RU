//
// Created by xiaowentao on 2024/2/15.
//

#ifndef GRAPH_SEARCH_BASE_GRAPH_H
#define GRAPH_SEARCH_BASE_GRAPH_H

#include <iostream>
#include <vector>

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

#include "point.h"
#include "node.h"
#include "util.h"

namespace Graph{
    using namespace util;

    class BaseGraphCluster{
    public:
        BaseGraphCluster();
        //根据分配的聚类簇号和node点集合构建一个方向图,核心
        //默认node中第一个点为聚簇中心点
        BaseGraphCluster(unsigned short cluster_id , std::vector <std::shared_ptr<Node> >& node);
        void GetFurthestDis(std::vector <std::shared_ptr<Node> >& node);
        unsigned short GetClusterId() const;
    private:
        std::vector <std::shared_ptr<Node>> graph_;
        std::unordered_map<ull,size_t> graph_node_table_;    // get the index by the node_id
        unsigned short cluster_id_{util::short_max};
    };


    class BaseGraph{
    public:
        BaseGraph();
        BaseGraph(std::vector<Point> data,size_t cluster_category_id);

    private:
        std::vector <BaseGraphCluster> base_graph_cluster_;
    };


}



#endif //GRAPH_SEARCH_BASE_GRAPH_H
