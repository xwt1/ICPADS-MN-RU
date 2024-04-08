//
// Created by root on 3/18/24.
//

#include <evaluate/evaluate.h>
#include <cmath>


template<class dist_t>
dist_t Evaluate::evaluateWithMMR(const std::vector <std::vector <dist_t> > &v,
                                 const std::vector <dist_t> q,
                                 dist_t lambda,
                                 hnswlib::Metric relevance_metric,
                                 hnswlib::Metric diversity_metric){
    auto sum  = (dist_t) 0;
    size_t siz = v.size();
    // calculate relevance
    for(auto i = 0 ; i < siz; i++){
        size_t dim1 = v[i].size();
        switch (relevance_metric) {
            case hnswlib::distance:{
                sum += calculateDistance(v[i],q,dim1);
            }
            break;
            case hnswlib::ip:{
                sum += calculateDistanceWithDotProduct(v[i],q,dim1);
            }break;
            default:{
                std::cerr<<"Wrong"<<std::endl;
                return -23333;
            }
        }
    }
    switch (relevance_metric) {
        case hnswlib::distance:{
            sum = - ((sum * lambda)/siz);
        }
        break;
        case hnswlib::ip:{
            sum = ((sum * lambda)/siz);
        }break;
    }

    dist_t mn = std::numeric_limits<dist_t>::max();
    for(auto i = 0 ; i < siz; i++){
        for(auto j =i+1 ;j < siz ;j++){
            size_t dim1 = v[i].size();
            size_t dim2 = v[j].size();
            if(dim1 != dim2) {
                std::cout<<"Wrong"<<std::endl;
                return -23333;
            }
            switch (diversity_metric) {
                case hnswlib::distance:{
                    dist_t temp = calculateDistance(v[i],v[j],dim1);
                    mn = std::min(mn,temp);
                }
                    break;
                case hnswlib::ip:{
                    dist_t temp = calculateDistanceWithDotProduct(v[i],v[j],dim1);
                    mn= std::min(mn,temp);
                }break;
            }
        }
    }
    if(diversity_metric == hnswlib::ip){
        mn = -mn;
    }
    mn = (1- lambda) * mn;
    return sum + mn;

//    dist_t mx = -std::numeric_limits<dist_t>::max();
//    for(auto i = 0 ; i < siz; i++){
//        for(auto j =i+1 ;j < siz ;j++){
//            size_t dim1 = v[i].size();
//            size_t dim2 = v[j].size();
//            if(dim1 != dim2) {
//                std::cout<<"Wrong"<<std::endl;
//                return -23333;
//            }
//            switch (diversity_metric) {
//                case hnswlib::distance:{
//                    dist_t temp = calculateDistance(v[i],v[j],dim1);
//                    mx = std::max(mx,temp);
//                }
//                break;
//                case hnswlib::ip:{
//                    dist_t temp = calculateDistanceWithDotProduct(v[i],v[j],dim1);
//                    mx = std::max(mx,temp);
//                }break;
//            }
//        }
//    }
//    mx = (1- lambda) * mx;
//    return sum + mx;
}

template<class dist_t>
dist_t Evaluate::evaluateWithILAD(const std::vector<std::vector<dist_t>>& v,
                               hnswlib::Metric diversity_metric){
    if (v.empty() || v[0].empty()) return 0.0;

    dist_t totalDistance = 0.0;
    size_t count = 0;
    size_t siz = v.size();

    for (size_t i = 0; i < siz; ++i) {
        for (size_t j = i + 1; j < siz; ++j) {
            totalDistance += calculateDistance(v[i], v[j], siz);
            ++count;
        }
    }

    // 避免除以零的情况
    if (count == 0) return 0.0;

    return totalDistance / static_cast<dist_t>(count);
}

template<class dist_t>
dist_t Evaluate::evaluateWithILMD(const std::vector<std::vector<dist_t>>& v,
                                  hnswlib::Metric diversity_metric){
    if (v.empty() || v[0].empty()) return 0.0;

    dist_t minDistance = std::numeric_limits<dist_t>::max();

    size_t siz = v.size();
    for (size_t i = 0; i < siz; ++i) {
        for (size_t j = i + 1; j < siz; ++j) {
            dist_t currentDistance = calculateDistance(v[i], v[j], siz);
            if (currentDistance < minDistance) {
                minDistance = currentDistance;
            }
        }
    }

    // 如果没有计算任何距离（例如，只有一个向量），则返回 0
    if (minDistance == std::numeric_limits<dist_t>::max()) return 0.0;

    return minDistance;
}


template<class dist_t>
dist_t Evaluate::evaluateWithDPP(const std::vector <std::vector <dist_t> > &v,const std::vector <dist_t> q,dist_t lambda,hnswlib::Metric relevance_metric,
                                 hnswlib::Metric diversity_metric){
    size_t v_siz = v.size();
    size_t dim = 0;
    if(!v.empty()){
        dim = v[0].size();
    }else{
        std::cout<<"Wrong"<<std::endl;
        return -23333;
    }
    Eigen::MatrixXd S(v_siz,v_siz);
    for(size_t i = 1; i < v_siz ;i++){
        for(size_t j = 0 ; j <= i;j++){
            S(i,j) = S(j,i) = calculateDistance(v[i],v[j],dim);
        }
    }

}


// 显式实例化
template float Evaluate::evaluateWithMMR<float>(const std::vector<std::vector<float>>&, const std::vector <float> ,float,hnswlib::Metric,hnswlib::Metric);
template float Evaluate::evaluateWithILAD<float>(const std::vector<std::vector<float>>&, hnswlib::Metric);
template float Evaluate::evaluateWithILMD<float>(const std::vector<std::vector<float>>&, hnswlib::Metric);