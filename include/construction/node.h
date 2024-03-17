//
// Created by xiaowentao on 2024/2/16.
//

#ifndef GRAPH_SEARCH_NODE_H
#define GRAPH_SEARCH_NODE_H

#include "util.h"
#include "point.h"

namespace Graph{
    class Node{
    public:
        util::ull GetNodeId() const;
//        unsigned short GetClusterId() const;
        const std::vector<std::shared_ptr<Node>>& GetEdges() const;

//        void SetClusterId(unsigned short cluster_id);
        void AddEdge(util::ull node_id, std::unordered_map<util::ull,std::shared_ptr<Node>> & node_table);
        void AddEdge(std::shared_ptr<Node> node);

    private:
        util::ull node_id_{util::ull_max};
//        unsigned short cluster_id_{util::short_max};
        std::vector <std::shared_ptr<Node> > edge_;
        Point coordinate;
    };


}




#endif //GRAPH_SEARCH_NODE_H
