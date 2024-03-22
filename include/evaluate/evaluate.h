//
// Created by root on 3/18/24.
//

#ifndef GRAPH_SEARCH_EVALUATE_H
#define GRAPH_SEARCH_EVALUATE_H

#include <iostream>
#include <vector>
#include "hnswlib/hnswlib.h"


class Evaluate{
public:
    template<class dist_t>
    static dist_t evaluateWithMMR(const std::vector <std::vector <dist_t> > &v,const std::vector <dist_t> q,dist_t lambda);

    template<class dist_t>
    static std::vector<std::vector<dist_t>> obtainRawMMR(const std::vector <std::vector <dist_t> > &v,const std::vector <dist_t> q,dist_t lambda, size_t k){
        size_t siz = v.size();
        size_t dim = q.size();
        std::vector<std::vector<dist_t>> result;
        if(siz == 0 )    return result;
        else{
            if(v[0].size() != q.size()|| siz < k){
                std::cout<<"wrong1"<<std::endl;
                return result;
            }
        }
        dist_t mn = std::numeric_limits<dist_t>::max();
        size_t index = -1;
        std::vector <dist_t> v_q_dist(siz,0);
        std::vector <bool> visited(siz,false);
        for(size_t i = 0; i < siz ; i++){
            dist_t sum_sq_diff = 0; // 初始化累加器
            for(size_t j = 0; j<dim ;j++){
                sum_sq_diff += (v[i][j] - q[j])*(v[i][j] - q[j]); // 累加差的平方
            }
            v_q_dist[i] = sqrt(sum_sq_diff); // 对累加的差的平方和取平方根
            if(mn > v_q_dist[i]){
                mn = v_q_dist[i];
                index = i;
            }
        }
        if(index != -1){
            result.push_back(v[index]);
            visited[index] = true;
        }else{
            std::cout<<"wrong2"<<std::endl;
            return result;
        }
//        初始化多样性项
        std::vector <dist_t> diversity(siz,0);
        for(size_t i = 0 ; i < siz ; i++){
            if(!visited[i]){
//                diversity[i] = std::max(calculateDistance<dist_t>(v[index],v[i],dim));
                diversity[i] = calculateDistance<dist_t>(v[index],v[i],dim);
            }
        }
//        迭代计算
        while(result.size() < k){
            dist_t mx = -std::numeric_limits<dist_t>::max();
            index = -1;
            for(size_t i = 0 ; i < siz ;i++){
                if(!visited[i]){
//                    选出一个新的点放入result
                    auto mmr_score = - lambda * v_q_dist[i] + (1- lambda)* diversity[i];
                    mx = std::max(mx,mmr_score);
                    index = i;
                }
            }
            if(index != -1){
                result.push_back(v[index]);
                visited[index] = true;
            }
//            更新mmr值
            for(size_t i = 0 ; i < siz ; i++){
                if(!visited[i]){
                    diversity[i] = std::max(calculateDistance<dist_t>(v[index],v[i],dim),diversity[i]);
                }
            }
        }
        return result;
    }
private:
    template<class dist_t>
    static dist_t calculateMMR(const std::vector<dist_t>& candidate, const std::vector<std::vector<dist_t>>& result, const std::vector<dist_t> q, dist_t lambda, size_t dim) {
        dist_t relevance = calculateDistance(candidate, q, dim); // 计算与q的距离作为相关性
        dist_t diversity = 0;
        for (const auto& selected : result) {
            diversity += calculateDistance(candidate, selected, dim); // 计算多样性，即与result中元素的距离
        }
        if (!result.empty()) {
            diversity /= result.size(); // 平均多样性
        }
        return lambda * relevance - (1 - lambda) * diversity; // MMR公式，lambda调节相关性与多样性的平衡
    }

    template<class dist_t>
    static dist_t calculateDistance(const std::vector<dist_t>& a, const std::vector<dist_t>& b, size_t dim) {
        dist_t dist_sq_sum = 0;
        for (size_t j = 0; j < dim; ++j) {
            dist_sq_sum += (a[j] - b[j]) * (a[j] - b[j]);
        }
        return sqrt(dist_sq_sum);
    }
};



#endif //GRAPH_SEARCH_EVALUATE_H
