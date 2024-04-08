//
// Created by root on 3/18/24.
//

#ifndef GRAPH_SEARCH_EVALUATE_H
#define GRAPH_SEARCH_EVALUATE_H

#include <iostream>
#include <vector>
#include "hnswlib/hnswlib.h"
#include <Eigen/Dense>


class Evaluate{
public:
//    enum Metric{
//        distance,
//        ip
//    };
    template<class dist_t>
    static dist_t evaluateWithMMR(const std::vector <std::vector <dist_t> > &v,
                                  const std::vector <dist_t> q,
                                  dist_t lambda,
                                  hnswlib::Metric relevance_metric = hnswlib::distance,
                                  hnswlib::Metric diversity_metric= hnswlib::distance);

    template<class dist_t>
    static dist_t evaluateWithILAD(const std::vector<std::vector<dist_t>>& v,
                                   hnswlib::Metric diversity_metric= hnswlib::distance);

    template<class dist_t>
    static dist_t evaluateWithILMD(const std::vector<std::vector<dist_t>>& v,
                                   hnswlib::Metric diversity_metric= hnswlib::distance);

    template<class dist_t>
    static dist_t evaluateWithDPP(const std::vector <std::vector <dist_t> > &v,const std::vector <dist_t> q,dist_t lambda,hnswlib::Metric relevance_metric = hnswlib::distance,
                                  hnswlib::Metric diversity_metric= hnswlib::distance);

    template<class dist_t>
    static std::vector<std::vector<dist_t>> obtainRawMMR(const std::vector <std::vector <dist_t> > &v,
                                                         const std::vector <dist_t> q,
                                                         dist_t lambda, size_t k,
                                                         hnswlib::Metric relevance_metric = hnswlib::distance,
                                                         hnswlib::Metric diversity_metric= hnswlib::distance){
        size_t siz = v.size();
        size_t dim = q.size();
        std::vector<std::vector<dist_t>> result;
        if(siz == 0 )    return result;
        else{
            if(v[0].size() != q.size()|| siz < k){
                std::cerr<<"wrong1"<<std::endl;
                return result;
            }
        }
//        dist_t mn = std::numeric_limits<dist_t>::max();
        size_t index = -1;
//        v_q_dist是v与q的距离
        std::vector <dist_t> v_q_dist(siz,0);
        std::vector <bool> visited(siz,false);
        switch (relevance_metric) {
            case hnswlib::distance:{
                dist_t mn = std::numeric_limits<dist_t>::max();
                for(size_t i = 0; i < siz ; i++){
                    v_q_dist[i] += calculateDistance<dist_t>(v[i],q,dim);
                    if(mn > v_q_dist[i]){
                        mn = v_q_dist[i];
                        index = i;
                    }
                }
//                std::cout<<mn<<std::endl;
            }break;
            case hnswlib::ip:{
                dist_t mx = -std::numeric_limits<dist_t>::max();
                for(size_t i = 0; i < siz ; i++){
                    v_q_dist[i] += calculateDistanceWithDotProduct<dist_t>(v[i],q,dim);
                    if(mx < v_q_dist[i]){
                        mx = v_q_dist[i];
                        index = i;
                    }
                }
            }break;
            default:{
                std::cerr<<"wrong1"<<std::endl;
                return result;
            }
        }

//        for(size_t i = 0; i < siz ; i++){
//            switch (relevance_metric) {
//                case hnswlib::distance:{
//                    v_q_dist[i] += -calculateDistance<dist_t>(v[i],q,dim);
//                }break;
//                case hnswlib::ip:{
//                    // 这里只是为了迎合求mn才加的负号,越小的记录说明点积越大,越相似
//                    v_q_dist[i] += calculateDistanceWithDotProduct<dist_t>(v[i],q,dim);
//                }break;
//                default:{
//                    std::cerr<<"wrong1"<<std::endl;
//                    return result;
//                }
//            }
//            if(mn > v_q_dist[i]){
//                mn = v_q_dist[i];
//                index = i;
//            }
//        }
        if(index != -1){
            result.push_back(v[index]);
            visited[index] = true;
        }else{
            std::cerr<<"wrong1"<<std::endl;
            return result;
        }
//        初始化多样性项
        std::vector <dist_t> diversity(siz,0);
        for(size_t i = 0 ; i < siz ; i++){
            if(!visited[i]){
                switch (diversity_metric) {
                    case hnswlib::distance:{
                        diversity[i] = calculateDistance<dist_t>(v[index],v[i],dim);
                    }break;
                    case hnswlib::ip:{
                        diversity[i] = calculateDistanceWithDotProduct<dist_t>(v[index],v[i],dim);
                    }break;
                }
            }
        }
//        迭代计算
        while(result.size() < k){
            dist_t mx = -std::numeric_limits<dist_t>::max();
            index = -1;
            for(size_t i = 0 ; i < siz ;i++){
                if(!visited[i]){
//                    选出一个新的点放入result
                    dist_t mmr_score = -23333;
                    switch(relevance_metric){
                        case hnswlib::distance:{
                            switch (diversity_metric) {
                                case hnswlib::distance:{
                                    mmr_score = -lambda * v_q_dist[i] + (1- lambda)* diversity[i];
                                }break;
                                case hnswlib::ip:{
                                    mmr_score = -lambda * v_q_dist[i] - (1- lambda)* diversity[i];
                                }break;
                            }
                        }break;
                        case hnswlib::ip:{
                            switch (diversity_metric) {
                                case hnswlib::distance:{
                                    mmr_score = lambda * v_q_dist[i] + (1- lambda)* diversity[i];
                                }break;
                                case hnswlib::ip:{
                                    mmr_score = lambda * v_q_dist[i] - (1- lambda)* diversity[i];
                                }break;
                            }
                        }break;
                    }
//                    auto mmr_score = - lambda * v_q_dist[i] + (1- lambda)* diversity[i];
                    mx = std::max(mx,mmr_score);
                    index = i;
                }
            }
            if(index != -1){
                result.push_back(v[index]);
                visited[index] = true;
            }
//            更新mmr值
//            for(size_t i = 0 ; i < siz ; i++){
//                if(!visited[i]){
//                    switch (diversity_metric) {
//                        case hnswlib::distance:{
//                            diversity[i] = std::max(calculateDistance<dist_t>(v[index],v[i],dim),diversity[i]);
//                        }break;
//                        case hnswlib::ip:{
//                            diversity[i] = std::max(calculateDistanceWithDotProduct(v[index],v[i],dim),diversity[i]);
//                        }break;
//                    }
//                }
//            }
            for(size_t i = 0 ; i < siz ; i++){
                if(!visited[i]){
                    switch (diversity_metric) {
                        case hnswlib::distance:{
                            diversity[i] = std::min(calculateDistance<dist_t>(v[index],v[i],dim),diversity[i]);
                        }break;
                        case hnswlib::ip:{
                            diversity[i] = std::min(calculateDistanceWithDotProduct(v[index],v[i],dim),diversity[i]);
                        }break;
                    }
                }
            }
        }
        return result;
    }
private:
    template<class dist_t>
    static dist_t calculateDistance(const std::vector<dist_t>& a, const std::vector<dist_t>& b, size_t dim) {
        dist_t dist_sq_sum = 0;
        for (size_t j = 0; j < dim; ++j) {
            dist_sq_sum += (a[j] - b[j]) * (a[j] - b[j]);
        }
        return sqrt(dist_sq_sum);
    }

    template<class dist_t>
    static dist_t calculateDistanceWithDotProduct(const std::vector<dist_t>& a, const std::vector<dist_t>& b, size_t dim) {
        dist_t dot_product = 0;
        for (size_t j = 0; j < dim; ++j) {
            dot_product += (a[j] * b[j]);
        }
        return dot_product;
    }
};



#endif //GRAPH_SEARCH_EVALUATE_H
