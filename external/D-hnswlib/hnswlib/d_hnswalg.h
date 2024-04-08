//
// Created by root on 4/4/24.
//

#pragma once

#include "hnswalg.h"

namespace hnswlib{
    static float
    cal_eu_dis(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return sqrt(res);
    }

    enum diversity_type{
        none,
        MMR
    };

    enum Metric{
        distance,
        ip
    };

    template<typename dist_t>
    class DHierarchicalNSW:public HierarchicalNSW<dist_t>{
    public:
        Metric relevance_metric_{distance};
        Metric diversity_metric_{distance};


        DHierarchicalNSW(SpaceInterface<dist_t> *s)
                : HierarchicalNSW<dist_t>(s) {
        }

        DHierarchicalNSW(
                SpaceInterface<dist_t> *s,
                const std::string &location,
                Metric relevance_metric,
                Metric diversity_metric,
                bool nmslib = false,
                size_t max_elements = 0,
                bool allow_replace_deleted = false)
                : HierarchicalNSW<dist_t>(s, location, nmslib, max_elements, allow_replace_deleted) {
            this->relevance_metric_ = relevance_metric;
            this->diversity_metric_ = diversity_metric;
        }

        DHierarchicalNSW(
                SpaceInterface<dist_t> *s,
                size_t max_elements,
                Metric relevance_metric,
                Metric diversity_metric,
                size_t M = 16,
                size_t ef_construction = 200,
                size_t random_seed = 100,
                bool allow_replace_deleted = false)
                : HierarchicalNSW<dist_t>(s, max_elements, M, ef_construction, random_seed, allow_replace_deleted) {
            this->relevance_metric_ = relevance_metric;
            this->diversity_metric_ = diversity_metric;
        }

        ~DHierarchicalNSW(){

        }
/*
         * count the diversity with Maximal
         *  diversity_search_result: the search result list, pair first is the relevance score, pair second is the node_id in hnsw
         *  candidate: candidate data for diversity_search_result
         *  diversity_metric: the metric to calculate diversity
         */
        auto countDiversityWithMin(std::vector<std::pair<dist_t, tableint>>& diversity_search_result,
                                   std::pair<dist_t, tableint> &candidate) const ->dist_t{


            char* candidate_data = this->getDataByInternalId(candidate.second);
            dist_t mn = std::numeric_limits<dist_t>::max();

//            {
//                for(size_t i = 0; i < diversity_search_result.size();i++){
//                    for(size_t j = i+1 ; j < diversity_search_result.size(); j ++){
//                        char * i_element = this->getDataByInternalId(diversity_search_result[i].second);
//                        char * j_element = this->getDataByInternalId(diversity_search_result[j].second);
//                        mx = std::max(mx,cal_eu_dis(i_element, j_element, this->dist_func_param_));
//                    }
//                }
//            }

            size_t res_siz = diversity_search_result.size();
            for(size_t  i = 0 ;i< res_siz; i++){
                char *top_element = this->getDataByInternalId(diversity_search_result[i].second);
                dist_t dist = -1;
                switch (diversity_metric_) {
                    case distance:{
                        dist = cal_eu_dis(candidate_data, top_element, this->dist_func_param_);
                    }break;
                    case ip:{
                        dist = InnerProductDistance(candidate_data, top_element, this->dist_func_param_);
                    }break;
                }
                mn = std::min(mn, dist);
            }
            return mn;
        }


        /*
         * count the diversity with Maximal
         *  diversity_search_result: the search result list, pair first is the relevance score, pair second is the node_id in hnsw
         *  candidate: candidate data for diversity_search_result
         *  diversity_metric: the metric to calculate diversity
         */
        auto countDiversityWithMax(std::vector<std::pair<dist_t, tableint>>& diversity_search_result,
                                   std::pair<dist_t, tableint> &candidate) const ->dist_t{


            char* candidate_data = this->getDataByInternalId(candidate.second);
            dist_t mx = -std::numeric_limits<dist_t>::max();

//            {
//                for(size_t i = 0; i < diversity_search_result.size();i++){
//                    for(size_t j = i+1 ; j < diversity_search_result.size(); j ++){
//                        char * i_element = this->getDataByInternalId(diversity_search_result[i].second);
//                        char * j_element = this->getDataByInternalId(diversity_search_result[j].second);
//                        mx = std::max(mx,cal_eu_dis(i_element, j_element, this->dist_func_param_));
//                    }
//                }
//            }

            size_t res_siz = diversity_search_result.size();
            for(size_t  i = 0 ;i< res_siz; i++){
                char *top_element = this->getDataByInternalId(diversity_search_result[i].second);
                dist_t dist = -1;
                switch (diversity_metric_) {
                    case distance:{
                        dist = cal_eu_dis(candidate_data, top_element, this->dist_func_param_);
                    }break;
                    case ip:{
                        // ip is not normally used for calculate diversity
                        dist = InnerProductDistance(candidate_data, top_element, this->dist_func_param_);
                    }break;
                }
                mx = std::max(mx, dist);
            }
            return mx;
        }

        /*
         * count candidate MMR score
         *  diversity_search_result: the search result list, pair first is the relevance score, pair second is the node_id in hnsw
         *  candidate: candidate data for diversity_search_result
         *  lambda: the tune parameter for relevance and diversity
         *  diversity_metric: the metric to calculate diversity(always distance)
         *  规律: 对相关性,使用距离度量为减号(距离越小,越邻近(相关),相关性得分越大),使用内积度量为加号(内积越大,越邻近(相关),相关性得分越大)
         *       对多样性,使用距离度量为加号(与已选集合的最大距离越大,说明离已选集合越远(越具有多样性),多样性得分越大),
         *               使用内积度量为减号(与已选集合的最大内积越小,说明离集合中最近的元素越远,多样性得分越大)
         */
        auto countMMR(std::vector<std::pair<dist_t, tableint>>& diversity_search_result,
                      std::pair<dist_t, tableint> &candidate,
                      dist_t lambda) const -> dist_t{
//            switch (relevance_metric_) {
//                case distance:{
//                    switch (diversity_metric_) {
//                        case distance:{
//                            return - lambda * candidate.first + (1 - lambda) * countDiversityWithMax(diversity_search_result,candidate);
//                        }break;
//                        case ip:{
//                            return - lambda * candidate.first - (1 - lambda) * countDiversityWithMax(diversity_search_result,candidate);
//                        }break;
//                    }
//                }break;
//                case ip:{
//                    switch (diversity_metric_) {
//                        case distance:{
//                            return lambda * candidate.first + (1 - lambda) * countDiversityWithMax(diversity_search_result,candidate);
//                        }break;
//                        case ip:{
//                            return lambda * candidate.first - (1 - lambda) * countDiversityWithMax(diversity_search_result,candidate);
//                        }break;
//                    }
//                }break;
//            }
            switch (relevance_metric_) {
                case distance:{
                    switch (diversity_metric_) {
                        case distance:{
                            return - lambda * candidate.first + (1 - lambda) * countDiversityWithMin(diversity_search_result,candidate);
                        }break;
                        case ip:{
                            return - lambda * candidate.first - (1 - lambda) * countDiversityWithMin(diversity_search_result,candidate);
                        }break;
                    }
                }break;
                case ip:{
                    switch (diversity_metric_) {
                        case distance:{
                            return lambda * candidate.first + (1 - lambda) * countDiversityWithMin(diversity_search_result,candidate);
                        }break;
                        case ip:{
                            return lambda * candidate.first - (1 - lambda) * countDiversityWithMin(diversity_search_result,candidate);
                        }break;
                    }
                }break;
            }
            std::cerr<<"wrong"<<std::endl;
            return -23333;
        }

        void initNewCandidate(std::vector<std::pair<dist_t, tableint>>& diversity_search_result,
                              std::list <std::pair<dist_t, tableint>> & candidate_set,
                              const void *data_point,
                              char *currObj1,
                              int candidate_id,
                              dist_t lambda) const{
            dist_t dist = -233333;
            switch (relevance_metric_) {
                case distance:{
                    dist = cal_eu_dis(data_point, currObj1, this->dist_func_param_);
                }break;
                case ip:{
                    dist = InnerProduct(data_point, currObj1, this->dist_func_param_);
                }break;
                default:{
                    std::cerr<<"wrong"<<std::endl;
                }
            }
            std::pair<dist_t, tableint> new_candidate = std::make_pair(dist,candidate_id);
            auto new_mmr = countMMR(diversity_search_result,new_candidate,lambda);
            //  将新邻居装入candidate_set
            candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
        }

        /*
         * choose the data for current turn
         *  diversity_search_result:
         *  candidate_set: pair first store the mmr score of the element in correspond index
         *                 pair second store the element id in correspond index
         */
        void chooseElementWithMMR(std::vector<std::pair<dist_t, tableint>>& diversity_search_result,
                                  std::list <std::pair<dist_t, tableint>> &candidate_set,
                                  const void *data_point,
                                  dist_t lambda,
                                  VisitedList * vl)const{
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            auto index = candidate_set.begin();
            dist_t mmr_dist = -std::numeric_limits<dist_t>::max();

            for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
                if(mmr_dist < it->first){
                    mmr_dist = it->first;
                    index = it;
                }
            }
            char *element_mmr = this->getDataByInternalId(index->second);
            // 这个地方用HNSW自己的度量函数计算一下就可以了,没必要写个switch判断一下,因为diversity_search_result第一个参数不在搜索过程中使用
            dist_t element_mmr_dist = this->fstdistfunc_(data_point, element_mmr, this->dist_func_param_);
            std::pair<dist_t, tableint> res_insert = std::make_pair(element_mmr_dist,index->second);
            // insert new data
            diversity_search_result.push_back(res_insert);
            // 更新candidate_set
            candidate_set.erase(index);
            
            // 更新原有的candidate_set的点集合,根据刚刚加入diversity_search_result的点更新
            for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
                char *temp = this->getDataByInternalId(it->second);
                switch (relevance_metric_) {
                    case distance:{
                        switch (diversity_metric_) {
                            case distance:{
                                //  mmr = argmax(-lambda* dist(Q,temp) + (1-lambda) * max(dist(d_i,temp)))

                                //  改成mmr = argmax(-lambda* dist(Q,temp) + (1-lambda) * min(dist(d_i,temp)))
                                dist_t relevance_score = lambda*cal_eu_dis(temp,data_point,this->dist_func_param_);
                                dist_t diversity_score = (1-lambda) * cal_eu_dis(temp, element_mmr, this->dist_func_param_);
                                dist_t last_diversity_score = it->first + relevance_score;
                                if(last_diversity_score > diversity_score){
                                    // 根据mmr公式,说明需要更新当前结点
                                    dist_t new_mmr =  - relevance_score +  diversity_score;
                                    it->first = new_mmr;
                                }
                            }break;
                            case ip:{
                                //  mmr = argmax(-lambda* dist(Q,temp) - (1-lambda) * max(ip(d_i,temp)))
                                dist_t relevance_score = lambda*cal_eu_dis(temp,data_point,this->dist_func_param_);
                                dist_t diversity_score = (1-lambda) * InnerProduct(temp, element_mmr, this->dist_func_param_);
                                dist_t last_diversity_score = -(it->first + relevance_score);
                                if(last_diversity_score > diversity_score){
                                    // 根据mmr公式,说明需要更新当前结点
                                    dist_t new_mmr =  - relevance_score -  diversity_score;
                                    it->first = new_mmr;
                                }
                            }break;
                        }
                    }break;
                    case ip:{
                        switch (diversity_metric_) {
                            case distance:{
                                //  mmr = argmax(lambda* ip(Q,temp) + (1-lambda) * max(dist(d_i,temp)))
                                //  改成 mmr = argmax(lambda* ip(Q,temp) + (1-lambda) * min(dist(d_i,temp)))
                                dist_t relevance_score = lambda*InnerProduct(temp,data_point,this->dist_func_param_);
                                dist_t diversity_score = (1-lambda) * cal_eu_dis(temp, element_mmr, this->dist_func_param_);
                                dist_t last_diversity_score = it->first - relevance_score;
                                if(last_diversity_score > diversity_score){
                                    // 根据mmr公式,说明需要更新当前结点
                                    dist_t new_mmr =  relevance_score +  diversity_score;
                                    it->first = new_mmr;
                                }
                            }break;
                            case ip:{
                                //  mmr = argmax(lambda* ip(Q,temp) - (1-lambda) * min(ip(d_i,temp)))
                                dist_t relevance_score = lambda*InnerProduct(temp,data_point,this->dist_func_param_);
                                dist_t diversity_score = (1-lambda) * InnerProduct(temp, element_mmr, this->dist_func_param_);
                                dist_t last_diversity_score = -(it->first - relevance_score);
                                if(last_diversity_score > diversity_score){
                                    // 根据mmr公式,说明需要更新当前结点
                                    dist_t new_mmr =  relevance_score -  diversity_score;
                                    it->first = new_mmr;
                                }
                            }break;
                        }
                    }break;
                }
            }
//        加入新邻居,增大可选集合
            int *neighbor = (int *) this->get_linklist0(diversity_search_result.back().second);
            size_t neighbor_size = this->getListCount((linklistsizeint*)neighbor);
            //  遍历每一个邻居
            for(size_t j = 1 ; j <= neighbor_size ; j++){
                int candidate_id = *(neighbor + j);
                if(visited_array[candidate_id] != visited_array_tag){
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (this->getDataByInternalId(candidate_id));
                    initNewCandidate(diversity_search_result,candidate_set,data_point,currObj1,candidate_id,lambda);

//                    dist_t dist = -233333;
//                    switch (relevance_metric_) {
//                        case distance:{
//                            dist = L2Sqr(data_point, currObj1, this->dist_func_param_);
//                        }break;
//                        case ip:{
//                            dist = InnerProduct(data_point, currObj1, this->dist_func_param_);
//                        }break;
//                        default:{
//                            std::cerr<<"wrong"<<std::endl;
//                        }
//                    }
//                    std::pair<dist_t, tableint> new_candidate = std::make_pair(dist,candidate_id);
//                    auto new_mmr = countMMR(diversity_search_result,new_candidate,lambda);
////                将新邻居装入candidate_set
//                    candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
                }
            }
        }

        void filterTopK(std::vector <std::pair<dist_t, tableint>> &top_candidate_vec,
                        std::vector<std::pair<dist_t, tableint>>& diversity_search_result,
                        dist_t lambda,
                        size_t k_1) const{
//        根据MMR将top_candidate_vec补齐至k_1 (todo)

        }

        void dealWithIp() const{

        }

        std::priority_queue<std::pair<dist_t, labeltype >> updateTopCandidateWithMMR(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> &top_candidates,
                const void *data_point,
                size_t k,
                dist_t lambda=0.8,
                size_t k_1 = 1,
                bool filter=false) const{
            std::vector<std::pair<dist_t, tableint>> diversity_search_result;
            std::list <std::pair<dist_t, tableint>> candidate_set;
            VisitedList * vl = this->visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::vector <std::pair<dist_t, tableint>> top_candidate_vec;
            while(!top_candidates.empty()){
                top_candidate_vec.push_back(top_candidates.top());
                top_candidates.pop();
            }
//            std::reverse(top_candidate_vec.begin(),top_candidate_vec.end());

//            if(!top_candidates.empty()){
//                auto tops = top_candidates.top();
//                diversity_search_result.push_back(tops);
//                visited_array[tops.second] = visited_array_tag;
//                top_candidates.pop();
//            }


//        选取出的k_1个点,最终会被放入top_candidate_with_mmr中
            if(filter){
//            如果选择过滤,就会利用MMR从top-K个顶点中选取出k_1个点作为多样性搜索的起始点
                filterTopK(top_candidate_vec,diversity_search_result,lambda,k_1);

            }else{

//            如果不选择过滤,就直接利用top-K个顶点中的前k_1个点作为多样性搜索的起始点
                while(!top_candidate_vec.empty() && diversity_search_result.size() < k_1 && diversity_search_result.size() < k){
                    auto tops = top_candidate_vec.back();
                    diversity_search_result.push_back(tops);
                    visited_array[tops.second] = visited_array_tag;
                    top_candidate_vec.pop_back();
                }


                while(!top_candidate_vec.empty()){
                    auto tops = top_candidate_vec.back();
                    int candidate_id = tops.second;
                    // 将剩余的原最近邻点加入候选集合
                    if(visited_array[candidate_id] != visited_array_tag) {
                        visited_array[candidate_id] = visited_array_tag;
                        char *currObj1 = (this->getDataByInternalId(candidate_id));
                        initNewCandidate(diversity_search_result,candidate_set,data_point,currObj1,candidate_id,lambda);
                    }
//                // 将剩余的原最近邻点的邻居加入候选集合
                    int *neighbor = (int *) this->get_linklist0(candidate_id);
                    size_t neighbor_size = this->getListCount((linklistsizeint*)neighbor);
                    for(size_t j = 1 ; j <= neighbor_size ; j++) {
                        int candidate_neighbour = *(neighbor + j);
                        if (visited_array[candidate_neighbour] != visited_array_tag) {
                            visited_array[candidate_neighbour] = visited_array_tag;
                            char *currObj1 = (this->getDataByInternalId(candidate_neighbour));
                            initNewCandidate(diversity_search_result, candidate_set, data_point, currObj1, candidate_neighbour,
                                             lambda);
                        }
                    }
                    top_candidate_vec.pop_back();
                }
            }



            //  现在diversity_search_result装着MMR公式中地S(已选择点)集合,现在访问每一个点的邻居点初始化candidate_set
            for(size_t i = 0;i< diversity_search_result.size() ; i++){
                int *neighbor = (int *) this->get_linklist0(diversity_search_result[i].second);
                size_t neighbor_size = this->getListCount((linklistsizeint*)neighbor);
//            遍历每一个邻居
                for(size_t j = 1 ; j <= neighbor_size ; j++){
                    int candidate_id = *(neighbor + j);
                    if(visited_array[candidate_id] != visited_array_tag){
                        visited_array[candidate_id] = visited_array_tag;
                        char *currObj1 = (this->getDataByInternalId(candidate_id));
                        initNewCandidate(diversity_search_result,candidate_set,data_point,currObj1,candidate_id,lambda);
//                        dist_t dist = -233333;
//                        switch (relevance_metric_) {
//                            case distance:{
//                                dist = L2Sqr(data_point, currObj1, this->dist_func_param_);
//                            }break;
//                            case ip:{
//                                dist = InnerProduct(data_point, currObj1, this->dist_func_param_);
//                            }break;
//                            default:{
//                                std::cerr<<"wrong"<<std::endl;
//                            }
//                        }
//                        std::pair<dist_t, tableint> new_candidate = std::make_pair(dist,candidate_id);
//                        auto new_mmr = countMMR(diversity_search_result,new_candidate,lambda);
////                将新邻居装入candidate_set
//                        candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
                    }
                }
            }



//            if(relevance_metric_ == ip){
//                std::vector <std::pair<dist_t, tableint>> temp_candidate(candidate_set.begin(),candidate_set.end());
//                size_t extras = 1;
//                size_t siz = temp_candidate.size();
//                size_t last_index = 0;
//                for(size_t extra = 0 ; extra < extras; extra ++){
//                    siz = temp_candidate.size();
//                    for(size_t i = last_index; i < siz; i++){
//                        int *neighbor = (int *) this->get_linklist0(temp_candidate[i].second);
//                        size_t neighbor_size = this->getListCount((linklistsizeint*)neighbor);
//                        for(size_t j = 1 ; j <= neighbor_size ; j++) {
//                            int candidate_id = *(neighbor + j);
//                            if (visited_array[candidate_id] != visited_array_tag) {
//                                visited_array[candidate_id] = visited_array_tag;
//                                char *currObj1 = (this->getDataByInternalId(candidate_id));
//                                dist_t dist = -233333;
//                                switch (relevance_metric_) {
//                                    case distance:{
//                                        dist = L2Sqr(data_point, currObj1, this->dist_func_param_);
//                                    }break;
//                                    case ip:{
//                                        dist = InnerProduct(data_point, currObj1, this->dist_func_param_);
//                                    }break;
//                                    default:{
//                                        std::cerr<<"wrong"<<std::endl;
//                                    }
//                                }
//                                std::pair<dist_t, tableint> new_candidate = std::make_pair(dist,candidate_id);
//                                auto new_mmr = countMMR(diversity_search_result,new_candidate,lambda);
//                                temp_candidate.emplace_back(std::make_pair(new_mmr,candidate_id));
//////                将新邻居装入candidate_set
////                        candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
//                            }
//                        }
//                    }
//                    last_index = siz;
//                }
//                candidate_set.assign(temp_candidate.begin(),temp_candidate.end());
//            }
            while(diversity_search_result.size() < k){
                chooseElementWithMMR(diversity_search_result,candidate_set,data_point,lambda,vl);
            }
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            for(int i = 0 ; i < diversity_search_result.size(); i++){
                auto items = diversity_search_result[i];
                result.push(std::make_pair(items.first,this->getExternalLabel(items.second)));
            }
            return result;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst>
        diversityAwareSearchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;
            if (this->cur_element_count == 0) return top_candidates;

//        currObj是当前搜到的入口点的id,一开始初始化成max_level层的一个点
            tableint currObj = this->enterpoint_node_;
//        curdist代表当前找到的每一层的入口点与query点的距离,在达到最底层之前,需要找到一个离query点最近的点
            dist_t curdist = this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);

            for (int level = this->maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

//                data存储当前currObj在该层的所有邻居
                    data = (unsigned int *) this->get_linklist(currObj, level);
                    int size = this->getListCount(data);

//                metric_hops和metric_distance_computations用于衡量HNSW的性能
                    this->metric_hops++;
                    this->metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > this->max_elements_)
                            throw std::runtime_error("cand error");
//                    看来fstdistfunc_是距离评判函数,后期可以更改这个以达成MMR的目的
                        dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(cand), this->dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }


            bool bare_bone_search = !this->num_deleted_ && !isIdAllowed;
            if (bare_bone_search) {
//                解决办法: https://cplusplus.com/forum/beginner/81030/
                top_candidates = HierarchicalNSW<dist_t>::template searchBaseLayerST<true>(
                        currObj, query_data, std::max(this->ef_, k), isIdAllowed);
            } else {
                top_candidates = HierarchicalNSW<dist_t>::template searchBaseLayerST<false>(
                        currObj, query_data, std::max(this->ef_, k), isIdAllowed);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
//        在这里插入代码,更新top_candidates
            return top_candidates;
        }


        std::priority_queue<std::pair<dist_t, labeltype >>
        D_searchKnn(const void *query_data, size_t k,diversity_type dt, size_t k_1 =1,dist_t lambda=0.8,BaseFilterFunctor* isIdAllowed = nullptr) const {
            switch (dt) {
                case none:{
                    return this->searchKnn(query_data,k,isIdAllowed);
                }
                    break;
                case MMR:{
//                    auto tops = [&](){
//                        switch (relevance_metric_) {
//                            case distance:{
//                                auto tops = diversityAwareSearchKnn(query_data,k_1,isIdAllowed);
//                                return tops;
//                            }break;
//                            case ip:{
//                                auto tops = diversityAwareSearchKnn(query_data,k,isIdAllowed);
//                                return tops;
//                            }break;
//                        }
//                    };
                    auto tops = diversityAwareSearchKnn(query_data,k_1,isIdAllowed);
//                    auto tops = diversityAwareSearchKnn(query_data,k,isIdAllowed);
//                    {
//                        while(!tops.empty()){
//                            std::cout<<tops.top().first<<" "<<tops.top().second<<std::endl;
//                            tops.pop();
//                        }
//                    }
                    auto ret = updateTopCandidateWithMMR(tops,query_data,k,lambda,k_1);
                    return ret;
                }
                    break;
            }
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            return result;
        }



//        /*
//         * count the diversity with Maximal
//         *      top_candidate_with_mmr:  the pointer to result candidate
//         *      result_size: top_candidate_with_mmr size
//         *      d_i: the element in candidate_set to calculate diversity with element in eps_for_diversity_search
//         */
//        auto countDiversityWithMax(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                   std::pair<dist_t, tableint> &d_i)const -> dist_t{
//            char* d_i_data = this->getDataByInternalId(d_i.second);
//            dist_t mx = -std::numeric_limits<dist_t>::max();
//            for(size_t  i = 0 ;i< eps_for_diversity_search.second; i++){
//                char *top_element = this->getDataByInternalId((eps_for_diversity_search.first+i)->second);
//                dist_t dist = L2Sqr(d_i_data, top_element, this->dist_func_param_);
//                mx = std::max(mx, dist);
//            }
//            return mx;
//        }
//
//        /*
//         * count the diversity with Centroid
//         *      centroid: center of the eps_for_diversity_search
//         *      d_i: the element in candidate_set to calculate diversity with element in eps_for_diversity_search
//         */
//        auto countDiverstyWithCentroid(std::pair<dist_t, tableint> &centroid,
//                                       std::pair<dist_t, tableint> &d_i)const ->dist_t{
//            char* d_i_data = getDataByInternalId(d_i.second);
//            char* centroid_data = getDataByInternalId(centroid.second);
//            dist_t dist = fstdistfunc_(d_i_data, centroid_data, this->dist_func_param_);
//            return dist;
//        }
//
//
//        void clearCandidateSet(std::list <std::pair<dist_t, tableint>> &candidate_set,
//                               VisitedList * vl,size_t k) const{
////        size_t can_size = candidate_set.size();
////        vl_type *visited_array = vl->mass;
////        vl_type visited_array_tag = vl->curV;
////        if(candidate_set.size() > k)    candidate_set.resize(k);
////        std::cout<<1<<std::endl;
//            if (k < candidate_set.size()) {
//                auto it = candidate_set.begin();
//                std::advance(it, k); // 将迭代器移动到第k个元素之后的位置
//                candidate_set.erase(it, candidate_set.end()); // 删除第k个元素之后的所有元素
//            }
//
////        for (auto it = candidate_set.begin(); it != candidate_set.end(); ) {
////            if (visited_array[it->second] == visited_array_tag) {
////                it = candidate_set.erase(it);
////            } else {
////                ++it;
////            }
////        }
//
////        std::vector <std::pair<dist_t, tableint>> temp;
////        temp.reserve(candidate_set.size());
////        for(size_t i = 0; i < can_size; i++){
////            if(candidate_set[i].second != visited_array_tag){
////                temp.emplace_back(candidate_set[i]);
////            }
////        }
////        candidate_set.clear();
////        for(auto &i : temp){
////            candidate_set.emplace_back(i);
////        }
//        }
//
//        /*
//         * count the Maximal marginal Relevance(MMR)
//         */
//        auto countMMR(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                      std::pair<dist_t, tableint> &d_i,
//                      dist_t lambda,
//                      bool relevance_metric = distance) const -> dist_t{
//            switch (relevance_metric) {
//                case distance:{
//                    return -lambda * d_i.first + (1 - lambda) * countDiversityWithMax(eps_for_diversity_search,d_i);
//                }break;
//                case ip:{
//                    return lambda * d_i.first + (1 - lambda) * countDiversityWithMax(eps_for_diversity_search,d_i);
//                }break;
//            }
//
//        }
//
////    auto count
//
//        /*
//         * count the Centroid Distance Maximization Enhanced MMR(CDM-EMMR)(形心距离最大化多样性增强MMR)
//         */
//        auto countCDM_EMMR(std::pair<dist_t, tableint> &centroid,
//                           std::pair<dist_t, tableint> &d_i,
//                           dist_t lambda){
//            return -lambda * d_i.first + (1 - lambda) * countDiverstyWithCentroid(centroid,d_i);
//        }
//
//
//
//        using high_resolution_clock = std::chrono::high_resolution_clock;
//        using time_point = std::chrono::time_point<high_resolution_clock>;
//        using duration = std::chrono::duration<double, std::micro>;
//
////    dist_t calculate_dist(){
////
////    }
//
////    void printEps(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search) const{
////        std::cout<<"eps vertex id"<<std::endl;
////        for(int i = 0;i < eps_for_diversity_search.second;i++){
////            std::cout<<(eps_for_diversity_search.first + i)->second<<" ";
////        }
////        std::cout<<std::endl;
////        std::cout<<"eps last elment vertex neighbour"<<std::endl;
////        int *neighbor = (int *) get_linklist0((eps_for_diversity_search.first+eps_for_diversity_search.second - 1)->second);
////        size_t neighbor_size = getListCount((linklistsizeint*)neighbor);
////        for(size_t j = 1 ; j <= neighbor_size ; j++){
////            int candidate_id = *(neighbor + j);
////            std::cout<<candidate_id<<" ";
////        }
////        std::cout<<std::endl;
////    }
////    void dfs(tableint now,std::vector <bool> &vis,int &count) const{
////        int *neighbor = (int *) get_linklist0(now);
////        size_t neighbor_size = getListCount((linklistsizeint*)neighbor);
////        for(size_t i = 1; i <= neighbor_size; i++){
////            int nei = *(neighbor + i);
////            if(!vis[nei]){
////                vis[nei] = true;
////                count++;
////                dfs(nei,vis,count);
////            }
////        }
////    }
//
////      使用这个函数时,candidate_set处于上一轮的状态
//        void chooseElementWithMMR(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                  std::list <std::pair<dist_t, tableint>> &candidate_set,
//                                  const void *data_point,
//                                  dist_t lambda,
//                                  VisitedList * vl,
//                                  Metric relevanceMetric = distance)const{
////        std::cout<<1234567<<std::endl;
//            vl_type *visited_array = vl->mass;
//            vl_type visited_array_tag = vl->curV;
//            auto index = candidate_set.begin();
//            dist_t mmr_dist = -std::numeric_limits<dist_t>::max();
//
//            for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//                if(mmr_dist < it->first){
//                    mmr_dist = it->first;
//                    index = it;
//                }
//            }
//            char *element_mmr = this->getDataByInternalId(index->second);
//            dist_t element_mmr_dist = this->fstdistfunc_(data_point, element_mmr, this->dist_func_param_);
//            std::pair<dist_t, tableint> e_insert = std::make_pair(element_mmr_dist,index->second);
////        向eps_for_diversity_search添加新点
//            insertEpsForDiversitySearch(eps_for_diversity_search,e_insert);
////        更新candidate_set
//            candidate_set.erase(index);
//
//
////        time_point start_time;
////        start_time = high_resolution_clock::now();
//
//            // 更新原有的candidate_set的点集合,根据刚刚加入eps_for_diversity_search的点更新
//            for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//                char *temp = this->getDataByInternalId(it->second);
//                dist_t temp_element_dist_with_lambda = (1-lambda) * L2Sqr(temp, element_mmr, this->dist_func_param_);
////            mmr = -lambda* dist(Q,temp) + (1-lambda) * max(dist(d_i,temp))
//                dist_t temp_query_dist = this->fstdistfunc_(temp,data_point,this->dist_func_param_);
//                dist_t res = it->first + lambda*temp_query_dist;    //(1-lambda) * max(dist(d_i,temp)) = mmr+ lambda* dist(Q,temp)
//                if(res < temp_element_dist_with_lambda){
////              根据MMR,说明需要更新当前结点
//                    dist_t new_mmr =  -lambda*temp_query_dist +  temp_element_dist_with_lambda;
//                    it->first = new_mmr;
//                }
//            }
//
////        auto stop_time = high_resolution_clock::now();
////        candidate_duration_ += stop_time - start_time;
////        std::cout<<123456789<<std::endl;
////        加入新邻居,增大可选集合
////        start_time = high_resolution_clock::now();
//
//
//            int *neighbor = (int *) this->get_linklist0((eps_for_diversity_search.first+eps_for_diversity_search.second - 1)->second);
//            size_t neighbor_size = this->getListCount((linklistsizeint*)neighbor);
////        neighbor_size = (neighbor_size > 5)?5:neighbor_size;
////            遍历每一个邻居
//            for(size_t j = 1 ; j <= neighbor_size ; j++){
//                int candidate_id = *(neighbor + j);
//                if(visited_array[candidate_id] != visited_array_tag){
//                    visited_array[candidate_id] = visited_array_tag;
//                    char *currObj1 = (this->getDataByInternalId(candidate_id));
//                    dist_t dist = this->fstdistfunc_(data_point, currObj1, this->dist_func_param_);
//                    std::pair<dist_t, tableint> d_i = std::make_pair(dist,candidate_id);
//                    auto new_mmr = countMMR(eps_for_diversity_search,d_i,lambda,relevanceMetric);
////                将新邻居装入candidate_set
//                    candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
//                }
//            }
//
////        stop_time = high_resolution_clock::now();
////        neighbour_duration_ += stop_time - start_time;
//        }
//
//        void updateCurrentCentroid(std::vector <std::vector<dist_t>> &centroid){
//
//        }
//
//        void chooseElementWithCDM_EMMR(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                       std::list <std::pair<dist_t, tableint>> &candidate_set,
//                                       std::vector <std::vector<dist_t>> &centroid,
//                                       const void *data_point,
//                                       dist_t lambda,
//                                       VisitedList * vl,
//                                       duration  &candidate_duration_,
//                                       duration  &neighbour_duration_)const{
////        1.从candidate_set中挑选一个最大的具有最大CDM_EMMR的点
//            vl_type *visited_array = vl->mass;
//            vl_type visited_array_tag = vl->curV;
//            auto index = candidate_set.begin();
//            dist_t mmr_dist = -std::numeric_limits<dist_t>::max();
//
//            for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//                if(mmr_dist < it->first){
//                    mmr_dist = it->first;
//                    index = it;
//                }
//            }
//            char *element_mmr = getDataByInternalId(index->second);
//            dist_t element_mmr_dist = fstdistfunc_(data_point, element_mmr, this->dist_func_param_);
//            std::pair<dist_t, tableint> e_insert = std::make_pair(element_mmr_dist,index->second);
////        向eps_for_diversity_search添加新点
//            insertEpsForDiversitySearch(eps_for_diversity_search,e_insert);
////        更新candidate_set
//            candidate_set.erase(index);
//
//
//        }
//
//        void insertEpsForDiversitySearch(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                         std::pair<dist_t, tableint> &e) const{
//            *(eps_for_diversity_search.first + eps_for_diversity_search.second) = e;
//            eps_for_diversity_search.second++;
//        }
//
////        void filterTopK(std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,typename HierarchicalNSW<dist_t>::CompareByFirst> &top_candidates,
////                        std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
////                        dist_t lambda,
////                        size_t k_1) const{
//////        根据MMR将top_candidates补齐至k_1 (todo)
////
////        }
//        /*
//         * method one to update result with MMR to gain diversity
//         *      top_candidates is the result to be updated
//         *      lambda is a parameter in MMR definition to control the ratio between similarity and diversity
//         *      k_1 is the entry point number for Diversity search,set to 1 by default
//         */
//        std::priority_queue<std::pair<dist_t, labeltype >> updateTopCandidateWithMMR(
//                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> &top_candidates,
//                const void *data_point,
//                size_t k,
//                dist_t lambda=0.8,
//                size_t k_1 = 1,
//                bool filter=false,
//                Metric relevanceMetric=distance) const{
////        在做完相似性搜索后,做进一步的多样性与相似性平衡的搜索来更新上一步的结果集合
//
////      为了效率,不使用智能指针了
////        std::shared_ptr<std::pair<dist_t, tableint> > eps_for_diversity_search(new std::pair<dist_t, tableint> [top_candidates.size()],
////                                                                             [](std::pair<dist_t, tableint>* p) { delete[] p; });
////      用普通指针记得释放first元素的空间
//            std::pair<std::pair<dist_t, tableint> *, size_t> eps_for_diversity_search = std::make_pair(new std::pair<dist_t, tableint> [k+1],0);
////        这里candidate_set中每一个元素,first存储的是MMR得分,second存储的是对应点的id
//            std::list <std::pair<dist_t, tableint>> candidate_set;
//            VisitedList * vl = this->visited_list_pool_->getFreeVisitedList();
//            vl_type *visited_array = vl->mass;
//            vl_type visited_array_tag = vl->curV;
//
//            if(!top_candidates.empty()){
//                auto tops = top_candidates.top();
//                insertEpsForDiversitySearch(eps_for_diversity_search,tops);
//                visited_array[tops.second] = visited_array_tag;
//                top_candidates.pop();
//            }
////        选取出的k_1个点,最终会被放入top_candidate_with_mmr中
//            if(filter){
////            如果选择过滤,就会利用MMR从top-K个顶点中选取出k_1个点作为多样性搜索的起始点
//                filterTopK(top_candidates,eps_for_diversity_search,lambda,k_1);
//            }else{
////            如果不选择过滤,就直接利用top-K个顶点中的前k_1个点作为多样性搜索的起始点
//                while(!top_candidates.empty() && eps_for_diversity_search.second < k_1){
//                    auto tops = top_candidates.top();
//                    insertEpsForDiversitySearch(eps_for_diversity_search,tops);
//                    visited_array[tops.second] = visited_array_tag;
//                    top_candidates.pop();
//                }
//            }
//            while(!top_candidates.empty())  top_candidates.pop();
//
////        std::cout<<123<<std::endl;
//
////        现在eps_for_diversity_search装着MMR公式中地S(已选择点)集合,现在访问每一个点的邻居点初始化candidate_set
//            for(size_t i = 0;i< eps_for_diversity_search.second ; i++){
//                int *neighbor = (int *) this->get_linklist0((eps_for_diversity_search.first+i)->second);
//                size_t neighbor_size = this->getListCount((linklistsizeint*)neighbor);
////            遍历每一个邻居
//                for(size_t j = 1 ; j <= neighbor_size ; j++){
//                    int candidate_id = *(neighbor + j);
//                    if(visited_array[candidate_id] != visited_array_tag){
//                        visited_array[candidate_id] = visited_array_tag;
//                        char *currObj1 = (this->getDataByInternalId(candidate_id));
//                        dist_t dist = this->fstdistfunc_(data_point, currObj1, this->dist_func_param_);
//                        std::pair<dist_t, tableint> d_i = std::make_pair(dist,candidate_id);
//                        auto new_mmr = countMMR(eps_for_diversity_search,d_i,lambda,relevanceMetric);
////                将新邻居装入candidate_set
//                        candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
//                    }
//                }
//            }
//
////        using namespace std::chrono; // 使用chrono命名空间，简化代码
////
////        auto start = high_resolution_clock::now(); // 开始时间
//
////        duration  candidate_duration_;
////        duration  neighbour_duration_;
//
////        std::cout<<12345<<std::endl;
//
////        {
////            std::cout<<" 17 vertex neighbour"<<std::endl;
////            int *neighbor = (int *) get_linklist0(17);
////            size_t neighbor_size = getListCount((linklistsizeint*)neighbor);
////            for(size_t j = 0 ; j < neighbor_size ; j++){
////                int candidate_id = *(neighbor + j);
////                std::cout<<candidate_id<<" ";
////            }
////            std::cout<<std::endl;
////        }
////        {
////            std::cout<<" 18 vertex neighbour"<<std::endl;
////            int *neighbor = (int *) get_linklist0(18);
////            size_t neighbor_size = getListCount((linklistsizeint*)neighbor);
////            for(size_t j = 0 ; j < neighbor_size ; j++){
////                int candidate_id = *(neighbor + j);
////                std::cout<<candidate_id<<" ";
////            }
////            std::cout<<std::endl;
////        }
//
////        printEps(eps_for_diversity_search);
////        迭代地根据MMR更新点集合
//            while(eps_for_diversity_search.second < k){
////            clearCandidateSet(candidate_set,vl,k);
////            std::cout<<"eps "<<eps_for_diversity_search.second<<std::endl;
////            std::cout<<"candi "<<candidate_set.size()<<std::endl;
//                chooseElementWithMMR(eps_for_diversity_search,candidate_set,data_point,lambda,vl,relevanceMetric);
////            printEps(eps_for_diversity_search);
//            }
////        std::cout<<1234567<<std::endl;
////        std::cout << "Time taken by function: "
////                  << candidate_duration_.count() << " microseconds" << std::endl;
////        std::cout << "Time taken by function: "
////                  << neighbour_duration_.count() << " microseconds" << std::endl;
////        auto stop = high_resolution_clock::now(); // 结束时间
//////
////        auto duration = duration_cast<microseconds>(stop - start); // 计算持续时间
//////
////        std::cout << "Time taken by function chooseElementWithMMR: "
////                  << duration.count() << " microseconds" << std::endl;
//
////        将起始点集合放入candidate_set
//
//
//            std::priority_queue<std::pair<dist_t, labeltype >> result;
//            for(int i = 0 ; i < eps_for_diversity_search.second; i++){
//                auto items = *(eps_for_diversity_search.first + i);
////            auto temp_item = ;
//                result.push(std::make_pair(items.first,this->getExternalLabel(items.second)));
////            result.push(std::pair<dist_t, labeltype>(,));
//            }
////        释放空间,现在candidate_set放入了起始的k_1个点
//            delete []eps_for_diversity_search.first;
//
//            return result;
//        }
//
////        using HierarchicalNSW<dist_t>::searchBaseLayerST;
//        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst>
//        diversityAwareSearchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
//            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;
//            if (this->cur_element_count == 0) return top_candidates;
//
////        currObj是当前搜到的入口点的id,一开始初始化成max_level层的一个点
//            tableint currObj = this->enterpoint_node_;
////        curdist代表当前找到的每一层的入口点与query点的距离,在达到最底层之前,需要找到一个离query点最近的点
//            dist_t curdist = this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);
//
//            for (int level = this->maxlevel_; level > 0; level--) {
//                bool changed = true;
//                while (changed) {
//                    changed = false;
//                    unsigned int *data;
//
////                data存储当前currObj在该层的所有邻居
//                    data = (unsigned int *) this->get_linklist(currObj, level);
//                    int size = this->getListCount(data);
//
////                metric_hops和metric_distance_computations用于衡量HNSW的性能
//                    this->metric_hops++;
//                    this->metric_distance_computations+=size;
//
//                    tableint *datal = (tableint *) (data + 1);
//                    for (int i = 0; i < size; i++) {
//                        tableint cand = datal[i];
//                        if (cand < 0 || cand > this->max_elements_)
//                            throw std::runtime_error("cand error");
////                    看来fstdistfunc_是距离评判函数,后期可以更改这个以达成MMR的目的
//                        dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(cand), this->dist_func_param_);
//
//                        if (d < curdist) {
//                            curdist = d;
//                            currObj = cand;
//                            changed = true;
//                        }
//                    }
//                }
//            }
//
//
//            bool bare_bone_search = !this->num_deleted_ && !isIdAllowed;
//            if (bare_bone_search) {
////                解决办法: https://cplusplus.com/forum/beginner/81030/
//                top_candidates = HierarchicalNSW<dist_t>::template searchBaseLayerST<true>(
//                        currObj, query_data, std::max(this->ef_, k), isIdAllowed);
//            } else {
//                top_candidates = HierarchicalNSW<dist_t>::template searchBaseLayerST<false>(
//                        currObj, query_data, std::max(this->ef_, k), isIdAllowed);
//            }
//
//            while (top_candidates.size() > k) {
//                top_candidates.pop();
//            }
////        在这里插入代码,更新top_candidates
//            return top_candidates;
//        }
//
//
//        std::priority_queue<std::pair<dist_t, labeltype >>
//        D_searchKnn(const void *query_data, size_t k,diversity_type dt,Metric relevanceMetric=distance, size_t k_1 =1,dist_t lambda=0.8,BaseFilterFunctor* isIdAllowed = nullptr) const {
//            switch (dt) {
//                case none:{
//                    return this->searchKnn(query_data,k,isIdAllowed);
//                }
//                    break;
//                case MMR:{
//                    using namespace std::chrono; // 使用chrono命名空间，简化代码
//                    auto tops = diversityAwareSearchKnn(query_data,k_1,isIdAllowed);
////                time_point start_time;
////                start_time = high_resolution_clock::now();
//                    auto ret = updateTopCandidateWithMMR(tops,query_data,k,lambda,k_1,relevanceMetric);
////                auto stop_time = high_resolution_clock::now();
////                auto durations = duration_cast<microseconds>(stop_time - start_time);
////                std::cout << "123Time taken by function: "
////                          << durations.count() << " microseconds" << std::endl;
//                    return ret;
//                }
//                    break;
//            }
//            std::priority_queue<std::pair<dist_t, labeltype >> result;
//            return result;
//        }
//
//        //      使用这个函数时,candidate_set处于上一轮的状态
//        void chooseElementWithMMRFromResult(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                            std::list <std::pair<dist_t, tableint>> &candidate_set,
//                                            const void *data_point,
//                                            dist_t lambda,
//                                            duration  &candidate_duration_,
//                                            duration  &neighbour_duration_)const{
//            auto index = candidate_set.begin();
//            dist_t mmr_dist = -std::numeric_limits<dist_t>::max();
//            for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//                if(mmr_dist < it->first){
//                    mmr_dist = it->first;
//                    index = it;
//                }
//            }
//            char *element_mmr = getDataByInternalId(index->second);
//            dist_t element_mmr_dist = fstdistfunc_(data_point, element_mmr, this->dist_func_param_);
//            std::pair<dist_t, tableint> e_insert = std::make_pair(element_mmr_dist,index->second);
////        向eps_for_diversity_search添加新点
//            insertEpsForDiversitySearch(eps_for_diversity_search,e_insert);
////        更新candidate_set
////        更新已有邻居
//            candidate_set.erase(index);
//
////        time_point start_time;
////        start_time = high_resolution_clock::now();
////
////        // 更新原有的candidate_set的点集合,根据刚刚加入eps_for_diversity_search的点更新
////        for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
////            char *temp = getDataByInternalId(it->second);
////            dist_t temp_element_dist_with_lambda = (1-lambda) * fstdistfunc_(temp, element_mmr, dist_func_param_);
//////            mmr = -lambda* dist(Q,temp) + (1-lambda) * max(dist(d_i,temp))
////            dist_t temp_query_dist = fstdistfunc_(temp,data_point,dist_func_param_);
////            dist_t res = it->first + lambda*temp_query_dist;    //(1-lambda) * max(dist(d_i,temp)) = mmr+ lambda* dist(Q,temp)
////            if(res < temp_element_dist_with_lambda){
//////              根据MMR,说明需要更新当前结点
////                dist_t new_mmr =  -lambda*temp_query_dist +  temp_element_dist_with_lambda;
////                it->first = new_mmr;
////            }
////        }
////
////        auto stop_time = high_resolution_clock::now();
////        candidate_duration_ += stop_time - start_time;
////
//////        加入新邻居,增大可选集合
////        start_time = high_resolution_clock::now();
////
////
////        int *neighbor = (int *) get_linklist0((eps_for_diversity_search.first+eps_for_diversity_search.second - 1)->second);
////        size_t neighbor_size = getListCount((linklistsizeint*)neighbor);
//////        neighbor_size = (neighbor_size > 5)?5:neighbor_size;
//////            遍历每一个邻居
////        for(size_t j = 0 ; j < neighbor_size ; j++){
////            int candidate_id = *(neighbor + j);
////            if(visited_array[candidate_id] != visited_array_tag){
////                visited_array[candidate_id] = visited_array_tag;
////                char *currObj1 = (getDataByInternalId(candidate_id));
////                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
////                std::pair<dist_t, tableint> d_i = std::make_pair(dist,candidate_id);
////                auto new_mmr = countMMR(eps_for_diversity_search,d_i,lambda);
//////                将新邻居装入candidate_set
////                candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
////            }
////        }
////
////        stop_time = high_resolution_clock::now();
////        neighbour_duration_ += stop_time - start_time;
//        }


    };
}