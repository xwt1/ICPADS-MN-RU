//
// Created by xiaowentao on 2024/2/16.
//

#ifndef GRAPH_SEARCH_UTIL_H
#define GRAPH_SEARCH_UTIL_H

#include <limits>

#include <mlpack/core.hpp>
#include "point.h"

namespace util{
    using ull = unsigned long long;

    constexpr unsigned short short_max = std::numeric_limits<unsigned short>::max();
    constexpr size_t size_t_max = std::numeric_limits<size_t>::max();
    constexpr ull ull_max = std::numeric_limits<ull>::max();



    class Util{
    public:
        static arma::mat ConvertPointsToArmaMat(const std::vector<Point>& points);
    };
}



#endif //GRAPH_SEARCH_UTIL_H
