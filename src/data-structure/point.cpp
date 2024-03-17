//
// Created by xiaowentao on 2024/2/15.
//

#include "point.h"
#include "point.h"

//#include "ut"
//#include "util.h"

Point::Point() {}

Point::Point(std::initializer_list<double> initList) : coordinates_(initList) {}

const std::vector<double>& Point::GetCoordinates() const {
    return coordinates_;
}

double Point::GetCoordinate(size_t index) const {
    if (index < coordinates_.size()) {
        return coordinates_[index];
    } else {
        throw std::out_of_range("Index is out of range.");
    }
}

size_t Point::GetDimension() const {
    return coordinates_.size();
}