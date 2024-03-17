//
// Created by xiaowentao on 2024/2/16.
//

#include <node.h>

// 获取节点ID
util::ull Graph::Node::GetNodeId() const {
    return node_id_;
}

//// 获取簇ID
//unsigned short Graph::Node::GetClusterId() const {
//    return cluster_id_;
//}

// 获取边的向量
const std::vector<std::shared_ptr<Graph::Node>>& Graph::Node::GetEdges() const {
    return edge_;
}

//// 设置簇ID
//void Graph::Node::SetClusterId(unsigned short cluster_id) {
//    cluster_id_ = cluster_id;
//}

// 添加一个边
void Graph::Node::AddEdge(util::ull node_id, std::unordered_map<util::ull,std::shared_ptr<Node>> & node_table) {
//    auto wtf = node_table[node_id];
    this->edge_.push_back(node_table[node_id]);
}

void Graph::Node::AddEdge(std::shared_ptr<Node> node) {
//    auto wtf = node_table[node_id];
    this->edge_.push_back(node);
}
