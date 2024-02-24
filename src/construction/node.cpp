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
void Graph::Node::AddEdge(std::shared_ptr<Node> edge) {
    edge_.push_back(edge);
}