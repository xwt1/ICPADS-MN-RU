//
// Created by xiaowentao on 2024/2/15.
//

#ifndef GRAPH_SEARCH_POINT_H
#define GRAPH_SEARCH_POINT_H

//#include "base_graph.h"


#include <vector>
#include <initializer_list>
#include <stdexcept>

class Point {
public:
    Point();
    Point(std::initializer_list<double> initList);

    const std::vector<double>& GetCoordinates() const;
    double GetCoordinate(size_t index) const;
    size_t GetDimension() const;

private:
    std::vector<double> coordinates; // 存储点的坐标
};

#endif //GRAPH_SEARCH_POINT_H
