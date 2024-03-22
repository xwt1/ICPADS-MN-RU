//
// Created by root on 3/18/24.
//

#include <evaluate/evaluate.h>
#include <cmath>


template<class dist_t>
dist_t Evaluate::evaluateWithMMR(const std::vector <std::vector <dist_t> > &v,const std::vector <dist_t> q,dist_t lambda){
    auto sum  = (dist_t) 0;
    size_t siz = v.size();
    for(auto i = 0 ; i < siz; i++){
        size_t dim1 = v[i].size();
        auto count_dis = [&](){
            dist_t temp = 0;
            for(auto k1 = 0; k1<dim1;k1++){
                temp+= ((v[i][k1])-(q[k1]))*((v[i][k1])-(q[k1]));
            }
            sum = sum + sqrt(temp);
        };
        count_dis();
    }
    sum = - ((sum * lambda)/siz);
    dist_t mx = -std::numeric_limits<dist_t>::max();
    for(auto i = 0 ; i < siz; i++){
        for(auto j =i+1 ;j < siz ;j++){
            size_t dim1 = v[i].size();
            size_t dim2 = v[j].size();
            if(dim1 != dim2) {
                std::cout<<"Wrong"<<std::endl;
                return -23333;
            }
            auto count_dis = [&](){
                dist_t temp = 0;
                for(auto k1 = 0; k1<dim1;k1++){
                    temp+= ((v[i][k1])-(v[j][k1]))*((v[i][k1])-(v[j][k1]));
                }
                temp = sqrt(temp);
                mx = std::max(mx,temp);
            };
            count_dis();
        }
    }
    mx = (1- lambda) * mx;
    return sum + mx;
}

// 显式实例化
template float Evaluate::evaluateWithMMR<float>(const std::vector<std::vector<float>>&, const std::vector <float> ,float);

