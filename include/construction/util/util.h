//
// Created by xiaowentao on 2024/2/16.
//

#ifndef GRAPH_SEARCH_UTIL_H
#define GRAPH_SEARCH_UTIL_H

#include <mlpack/core.hpp>
#include "point.h"

class Util{
public:
    static arma::mat ConvertPointsToArmaMat(const std::vector<Point>& points);
};


#endif //GRAPH_SEARCH_UTIL_H
