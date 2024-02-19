//
// Created by xiaowentao on 2024/2/15.
//

#ifndef GRAPH_SEARCH_POINT_H
#define GRAPH_SEARCH_POINT_H

#include <vector>


class Point {
public:
    std::vector<double> coordinates; // 存储点的坐标

    // 使用初始化列表构造函数接受任意数量的维度数据
    Point(std::initializer_list<double> initList) : coordinates(initList) {}

    template<class T>
    Point(){

    }
};

#endif //GRAPH_SEARCH_POINT_H
