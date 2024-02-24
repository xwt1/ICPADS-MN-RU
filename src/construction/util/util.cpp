//
// Created by xiaowentao on 2024/2/16.
//

#include "util.h"

arma::mat util::Util::ConvertPointsToArmaMat(const std::vector<Point>& points) {
    if (points.empty()) return arma::mat(); // 如果输入为空，返回一个空矩阵

    size_t dimensions = points[0].GetDimension(); // 假设所有点维度相同
    arma::mat data(dimensions, points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < dimensions; ++j) {
            data(j, i) = points[i].GetCoordinate(j);
        }
    }

    return data;
}