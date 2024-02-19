//
// Created by xiaowentao on 2024/2/15.
//

#ifndef GRAPH_SEARCH_BASE_GRAPH_H
#define GRAPH_SEARCH_BASE_GRAPH_H

#include <iostream>
#include <vector>

#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>

#include "point.h"
#include "util.h"

namespace Graph{
//    template <class T>
    class BaseGraph{
    public:
        BaseGraph();
//        BaseGraph(std::vector<T> data);
        BaseGraph(std::vector<Point> data);
    };
}



#endif //GRAPH_SEARCH_BASE_GRAPH_H
