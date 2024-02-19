//
// Created by xiaowentao on 2024/2/15.
//

#include "data-structure/point.h"

//#include "ut"
#include "util.h"

Point::Point() {}

Point::Point(std::initializer_list<double> initList) : coordinates(initList) {}

const std::vector<double>& Point::GetCoordinates() const {
    return coordinates;
}

double Point::GetCoordinate(size_t index) const {
    if (index < coordinates.size()) {
        return coordinates[index];
    } else {
        throw std::out_of_range("Index is out of range.");
    }
}

size_t Point::GetDimension() const {
    return coordinates.size();
}