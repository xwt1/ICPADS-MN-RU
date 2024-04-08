#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <cmath>
#include <list>
#include <memory>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;


template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }

    //构造函数一: 可以加载已经构建好的索引
    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }

    // 构造函数2: 根据所给的空间构造
    // s是给定好的度量空间,一般来说都是l2欧式空间
    // M是每个点需要与其余点建立的连接数量
    // ef_construction: 搜索缓冲池的大小
    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
//        利用仿函数,由于STL优先队列默认是大根堆,所以默认比较函数采用的是LESS规则
//        即若比较函数返回true,则优先队列认为第一个参数小于第二个参数(即符合LESS小于的意思)
//        如果返回的是false,则优先队列认为第二个参数小于第一个参数
//        这里利用其第一个参数作为比较参数,如果第一个参数小于第二个参数(返回值为true),
//        那么就会把第二个参数排在更靠近堆顶,由于HNSW保存的是相反数,故保存的距离值越大,实际距离越小,离query越近.
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
//    collect_metrics表示是否收集数据
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
//        已访问结点集合初始化
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

//        candidate
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
//        这里获取距离的下限,如果加了终止条件等等,估计HNSW会根据入口点与query点的距离作为阈值,在搜索过程中做一些停止条件的判断
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
//            如果满足了上面的条件,说明允许急停并且该点允许做这种操作,则把距离的下限设置成入口点与query点的距离
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
//            不满足，则把下限设置成无穷小
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

//            判断一下提前终止条件是否成立
            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif
//            遍历每一个出边
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

//                    candidate_id是候选点的邻居结点,是更新top_candidates和candidate_set的关键

//                    这里要根据一些预设条件判断一下这个邻居到底能不能作为候选点
                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

//                    如果能作为候选点,就尝试更新两个优先队列
                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
//                            这里是搜索函数更新的关键,可以在这个地方设置timestamp更新结果集合

                            top_candidates.emplace(dist, candidate_id);

                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
//        update写在visited_list_pool_->releaseVisitedList(vl)下面有好处,较大概率重复利用刚才的内存空间
        //        在这里利用MMR更新top_candidates,利用MMR和top_candidates做MMR搜索
//                            this->updateTopCandidateWithMMR(top_candidates,data,3);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            // 如果不是删除操作,进入添点阶段
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

//        currObj是当前搜到的入口点的id,一开始初始化成max_level层的一个点
        tableint currObj = enterpoint_node_;
//        curdist代表当前找到的每一层的入口点与query点的距离,在达到最底层之前,需要找到一个离query点最近的点
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

//                data存储当前currObj在该层的所有邻居
                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);

//                metric_hops和metric_distance_computations用于衡量HNSW的性能
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
//                    看来fstdistfunc_是距离评判函数,后期可以更改这个以达成MMR的目的
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

//    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//    diversityAwareSearchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
//        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//        if (cur_element_count == 0) return top_candidates;
//
////        currObj是当前搜到的入口点的id,一开始初始化成max_level层的一个点
//        tableint currObj = enterpoint_node_;
////        curdist代表当前找到的每一层的入口点与query点的距离,在达到最底层之前,需要找到一个离query点最近的点
//        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
//
//        for (int level = maxlevel_; level > 0; level--) {
//            bool changed = true;
//            while (changed) {
//                changed = false;
//                unsigned int *data;
//
////                data存储当前currObj在该层的所有邻居
//                data = (unsigned int *) get_linklist(currObj, level);
//                int size = getListCount(data);
//
////                metric_hops和metric_distance_computations用于衡量HNSW的性能
//                metric_hops++;
//                metric_distance_computations+=size;
//
//                tableint *datal = (tableint *) (data + 1);
//                for (int i = 0; i < size; i++) {
//                    tableint cand = datal[i];
//                    if (cand < 0 || cand > max_elements_)
//                        throw std::runtime_error("cand error");
////                    看来fstdistfunc_是距离评判函数,后期可以更改这个以达成MMR的目的
//                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
//
//                    if (d < curdist) {
//                        curdist = d;
//                        currObj = cand;
//                        changed = true;
//                    }
//                }
//            }
//        }
//
//
//        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
//        if (bare_bone_search) {
//            top_candidates = searchBaseLayerST<true>(
//                    currObj, query_data, std::max(ef_, k), isIdAllowed);
//        } else {
//            top_candidates = searchBaseLayerST<false>(
//                    currObj, query_data, std::max(ef_, k), isIdAllowed);
//        }
//
//        while (top_candidates.size() > k) {
//            top_candidates.pop();
//        }
////        在这里插入代码,更新top_candidates
//        return top_candidates;
//    }

    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

//    /*
//     * count the diversity with Maximal
//     *      top_candidate_with_mmr:  the pointer to result candidate
//     *      result_size: top_candidate_with_mmr size
//     *      d_i: the element in candidate_set to calculate diversity with element in eps_for_diversity_search
//     */
//    auto countDiversityWithMax(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                 std::pair<dist_t, tableint> &d_i)const -> dist_t{
//        char* d_i_data = getDataByInternalId(d_i.second);
//        dist_t mx = -std::numeric_limits<dist_t>::max();
//        for(size_t  i = 0 ;i< eps_for_diversity_search.second; i++){
//            char *top_element = getDataByInternalId((eps_for_diversity_search.first+i)->second);
//            dist_t dist = L2Sqr(d_i_data, top_element, dist_func_param_);
//            mx = std::max(mx, dist);
//        }
//        return mx;
//    }
//
//    /*
//     * count the diversity with Centroid
//     *      centroid: center of the eps_for_diversity_search
//     *      d_i: the element in candidate_set to calculate diversity with element in eps_for_diversity_search
//     */
//    auto countDiverstyWithCentroid(std::pair<dist_t, tableint> &centroid,
//                                   std::pair<dist_t, tableint> &d_i)const ->dist_t{
//        char* d_i_data = getDataByInternalId(d_i.second);
//        char* centroid_data = getDataByInternalId(centroid.second);
//        dist_t dist = fstdistfunc_(d_i_data, centroid_data, dist_func_param_);
//        return dist;
//    }
//
//
//    void clearCandidateSet(std::list <std::pair<dist_t, tableint>> &candidate_set,
//                           VisitedList * vl,size_t k) const{
////        size_t can_size = candidate_set.size();
////        vl_type *visited_array = vl->mass;
////        vl_type visited_array_tag = vl->curV;
////        if(candidate_set.size() > k)    candidate_set.resize(k);
////        std::cout<<1<<std::endl;
//        if (k < candidate_set.size()) {
//            auto it = candidate_set.begin();
//            std::advance(it, k); // 将迭代器移动到第k个元素之后的位置
//            candidate_set.erase(it, candidate_set.end()); // 删除第k个元素之后的所有元素
//        }
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
//    }
//
//    /*
//     * count the Maximal marginal Relevance(MMR)
//     */
//    auto countMMR(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                  std::pair<dist_t, tableint> &d_i,
//                  dist_t lambda,
//                  bool relevance_metric = distance) const -> dist_t{
//        switch (relevance_metric) {
//            case distance:{
//                return -lambda * d_i.first + (1 - lambda) * countDiversityWithMax(eps_for_diversity_search,d_i);
//            }break;
//            case ip:{
//                return lambda * d_i.first + (1 - lambda) * countDiversityWithMax(eps_for_diversity_search,d_i);
//            }break;
//        }
//
//    }
//
////    auto count
//
//    /*
//     * count the Centroid Distance Maximization Enhanced MMR(CDM-EMMR)(形心距离最大化多样性增强MMR)
//     */
//    auto countCDM_EMMR(std::pair<dist_t, tableint> &centroid,
//                  std::pair<dist_t, tableint> &d_i,
//                  dist_t lambda){
//        return -lambda * d_i.first + (1 - lambda) * countDiverstyWithCentroid(centroid,d_i);
//    }
//
//
//
//    using high_resolution_clock = std::chrono::high_resolution_clock;
//    using time_point = std::chrono::time_point<high_resolution_clock>;
//    using duration = std::chrono::duration<double, std::micro>;
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
//    void chooseElementWithMMR(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                              std::list <std::pair<dist_t, tableint>> &candidate_set,
//                              const void *data_point,
//                              dist_t lambda,
//                              VisitedList * vl,
//                              relevance_metric relevanceMetric = distance)const{
////        std::cout<<1234567<<std::endl;
//        vl_type *visited_array = vl->mass;
//        vl_type visited_array_tag = vl->curV;
//        auto index = candidate_set.begin();
//        dist_t mmr_dist = -std::numeric_limits<dist_t>::max();
//
//        for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//            if(mmr_dist < it->first){
//                mmr_dist = it->first;
//                index = it;
//            }
//        }
//        char *element_mmr = getDataByInternalId(index->second);
//        dist_t element_mmr_dist = fstdistfunc_(data_point, element_mmr, dist_func_param_);
//        std::pair<dist_t, tableint> e_insert = std::make_pair(element_mmr_dist,index->second);
////        向eps_for_diversity_search添加新点
//        insertEpsForDiversitySearch(eps_for_diversity_search,e_insert);
////        更新candidate_set
//        candidate_set.erase(index);
//
//
////        time_point start_time;
////        start_time = high_resolution_clock::now();
//
//        // 更新原有的candidate_set的点集合,根据刚刚加入eps_for_diversity_search的点更新
//        for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//            char *temp = getDataByInternalId(it->second);
//            dist_t temp_element_dist_with_lambda = (1-lambda) * L2Sqr(temp, element_mmr, dist_func_param_);
////            mmr = -lambda* dist(Q,temp) + (1-lambda) * max(dist(d_i,temp))
//            dist_t temp_query_dist = fstdistfunc_(temp,data_point,dist_func_param_);
//            dist_t res = it->first + lambda*temp_query_dist;    //(1-lambda) * max(dist(d_i,temp)) = mmr+ lambda* dist(Q,temp)
//            if(res < temp_element_dist_with_lambda){
////              根据MMR,说明需要更新当前结点
//                dist_t new_mmr =  -lambda*temp_query_dist +  temp_element_dist_with_lambda;
//                it->first = new_mmr;
//            }
//        }
//
////        auto stop_time = high_resolution_clock::now();
////        candidate_duration_ += stop_time - start_time;
////        std::cout<<123456789<<std::endl;
////        加入新邻居,增大可选集合
////        start_time = high_resolution_clock::now();
//
//
//        int *neighbor = (int *) get_linklist0((eps_for_diversity_search.first+eps_for_diversity_search.second - 1)->second);
//        size_t neighbor_size = getListCount((linklistsizeint*)neighbor);
////        neighbor_size = (neighbor_size > 5)?5:neighbor_size;
////            遍历每一个邻居
//        for(size_t j = 1 ; j <= neighbor_size ; j++){
//            int candidate_id = *(neighbor + j);
//            if(visited_array[candidate_id] != visited_array_tag){
//                visited_array[candidate_id] = visited_array_tag;
//                char *currObj1 = (getDataByInternalId(candidate_id));
//                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
//                std::pair<dist_t, tableint> d_i = std::make_pair(dist,candidate_id);
//                auto new_mmr = countMMR(eps_for_diversity_search,d_i,lambda,relevanceMetric);
////                将新邻居装入candidate_set
//                candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
//            }
//        }
//
////        stop_time = high_resolution_clock::now();
////        neighbour_duration_ += stop_time - start_time;
//    }
//
//    void updateCurrentCentroid(std::vector <std::vector<dist_t>> &centroid
//                               ){
//
//    }
//
//    void chooseElementWithCDM_EMMR(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                   std::list <std::pair<dist_t, tableint>> &candidate_set,
//                                   std::vector <std::vector<dist_t>> &centroid,
//                                   const void *data_point,
//                                   dist_t lambda,
//                                   VisitedList * vl,
//                                   duration  &candidate_duration_,
//                                   duration  &neighbour_duration_)const{
////        1.从candidate_set中挑选一个最大的具有最大CDM_EMMR的点
//        vl_type *visited_array = vl->mass;
//        vl_type visited_array_tag = vl->curV;
//        auto index = candidate_set.begin();
//        dist_t mmr_dist = -std::numeric_limits<dist_t>::max();
//
//        for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//            if(mmr_dist < it->first){
//                mmr_dist = it->first;
//                index = it;
//            }
//        }
//        char *element_mmr = getDataByInternalId(index->second);
//        dist_t element_mmr_dist = fstdistfunc_(data_point, element_mmr, dist_func_param_);
//        std::pair<dist_t, tableint> e_insert = std::make_pair(element_mmr_dist,index->second);
////        向eps_for_diversity_search添加新点
//        insertEpsForDiversitySearch(eps_for_diversity_search,e_insert);
////        更新candidate_set
//        candidate_set.erase(index);
//
//
//    }
//
//    void insertEpsForDiversitySearch(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                                     std::pair<dist_t, tableint> &e) const{
//        *(eps_for_diversity_search.first + eps_for_diversity_search.second) = e;
//        eps_for_diversity_search.second++;
//    }
//
//    void filterTopK(std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
//                    std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                    dist_t lambda,
//                    size_t k_1) const{
////        根据MMR将top_candidates补齐至k_1 (todo)
//
//    }
//    /*
//     * method one to update result with MMR to gain diversity
//     *      top_candidates is the result to be updated
//     *      lambda is a parameter in MMR definition to control the ratio between similarity and diversity
//     *      k_1 is the entry point number for Diversity search,set to 1 by default
//     */
//    std::priority_queue<std::pair<dist_t, labeltype >> updateTopCandidateWithMMR(
//            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
//            const void *data_point,
//            size_t k,
//            dist_t lambda=0.8,
//            size_t k_1 = 1,
//            bool filter=false,
//            relevance_metric relevanceMetric=distance) const{
////        在做完相似性搜索后,做进一步的多样性与相似性平衡的搜索来更新上一步的结果集合
//
////      为了效率,不使用智能指针了
////        std::shared_ptr<std::pair<dist_t, tableint> > eps_for_diversity_search(new std::pair<dist_t, tableint> [top_candidates.size()],
////                                                                             [](std::pair<dist_t, tableint>* p) { delete[] p; });
////      用普通指针记得释放first元素的空间
//        std::pair<std::pair<dist_t, tableint> *, size_t> eps_for_diversity_search = std::make_pair(new std::pair<dist_t, tableint> [k+1],0);
////        这里candidate_set中每一个元素,first存储的是MMR得分,second存储的是对应点的id
//        std::list <std::pair<dist_t, tableint>> candidate_set;
//        VisitedList * vl = visited_list_pool_->getFreeVisitedList();
//        vl_type *visited_array = vl->mass;
//        vl_type visited_array_tag = vl->curV;
//
//        if(!top_candidates.empty()){
//            auto tops = top_candidates.top();
//            insertEpsForDiversitySearch(eps_for_diversity_search,tops);
//            visited_array[tops.second] = visited_array_tag;
//            top_candidates.pop();
//        }
////        选取出的k_1个点,最终会被放入top_candidate_with_mmr中
//        if(filter){
////            如果选择过滤,就会利用MMR从top-K个顶点中选取出k_1个点作为多样性搜索的起始点
//            filterTopK(top_candidates,eps_for_diversity_search,lambda,k_1);
//        }else{
////            如果不选择过滤,就直接利用top-K个顶点中的前k_1个点作为多样性搜索的起始点
//            while(!top_candidates.empty() && eps_for_diversity_search.second < k_1){
//                auto tops = top_candidates.top();
//                insertEpsForDiversitySearch(eps_for_diversity_search,tops);
//                visited_array[tops.second] = visited_array_tag;
//                top_candidates.pop();
//            }
//        }
//        while(!top_candidates.empty())  top_candidates.pop();
//
////        std::cout<<123<<std::endl;
//
////        现在eps_for_diversity_search装着MMR公式中地S(已选择点)集合,现在访问每一个点的邻居点初始化candidate_set
//        for(size_t i = 0;i< eps_for_diversity_search.second ; i++){
//            int *neighbor = (int *) get_linklist0((eps_for_diversity_search.first+i)->second);
//            size_t neighbor_size = getListCount((linklistsizeint*)neighbor);
////            遍历每一个邻居
//            for(size_t j = 1 ; j <= neighbor_size ; j++){
//                int candidate_id = *(neighbor + j);
//                if(visited_array[candidate_id] != visited_array_tag){
//                    visited_array[candidate_id] = visited_array_tag;
//                    char *currObj1 = (getDataByInternalId(candidate_id));
//                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
//                    std::pair<dist_t, tableint> d_i = std::make_pair(dist,candidate_id);
//                    auto new_mmr = countMMR(eps_for_diversity_search,d_i,lambda,relevanceMetric);
////                将新邻居装入candidate_set
//                    candidate_set.emplace_back(std::make_pair(new_mmr,candidate_id));
//                }
//            }
//        }
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
//        while(eps_for_diversity_search.second < k){
////            clearCandidateSet(candidate_set,vl,k);
////            std::cout<<"eps "<<eps_for_diversity_search.second<<std::endl;
////            std::cout<<"candi "<<candidate_set.size()<<std::endl;
//            chooseElementWithMMR(eps_for_diversity_search,candidate_set,data_point,lambda,vl,relevanceMetric);
////            printEps(eps_for_diversity_search);
//        }
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
//        std::priority_queue<std::pair<dist_t, labeltype >> result;
//        for(int i = 0 ; i < eps_for_diversity_search.second; i++){
//            auto items = *(eps_for_diversity_search.first + i);
////            auto temp_item = ;
//            result.push(std::make_pair(items.first,getExternalLabel(items.second)));
////            result.push(std::pair<dist_t, labeltype>(,));
//        }
////        释放空间,现在candidate_set放入了起始的k_1个点
//        delete []eps_for_diversity_search.first;
//
//        return result;
//    }
//
//    std::priority_queue<std::pair<dist_t, labeltype >>
//    D_searchKnn(const void *query_data, size_t k,diversity_type dt,relevance_metric relevanceMetric=distance, size_t k_1 =1,dist_t lambda=0.8,BaseFilterFunctor* isIdAllowed = nullptr) const {
//        switch (dt) {
//            case none:{
//                return searchKnn(query_data,k,isIdAllowed);
//            }
//            break;
//            case MMR:{
//                using namespace std::chrono; // 使用chrono命名空间，简化代码
//                auto tops = diversityAwareSearchKnn(query_data,k_1,isIdAllowed);
////                time_point start_time;
////                start_time = high_resolution_clock::now();
//                auto ret = updateTopCandidateWithMMR(tops,query_data,k,lambda,k_1,relevanceMetric);
////                auto stop_time = high_resolution_clock::now();
////                auto durations = duration_cast<microseconds>(stop_time - start_time);
////                std::cout << "123Time taken by function: "
////                          << durations.count() << " microseconds" << std::endl;
//                return ret;
//            }
//            break;
//        }
//        std::priority_queue<std::pair<dist_t, labeltype >> result;
//        return result;
//    }
//
//    //      使用这个函数时,candidate_set处于上一轮的状态
//    void chooseElementWithMMRFromResult(std::pair<std::pair<dist_t, tableint> *, size_t>& eps_for_diversity_search,
//                              std::list <std::pair<dist_t, tableint>> &candidate_set,
//                              const void *data_point,
//                              dist_t lambda,
//                              duration  &candidate_duration_,
//                              duration  &neighbour_duration_)const{
//        auto index = candidate_set.begin();
//        dist_t mmr_dist = -std::numeric_limits<dist_t>::max();
//        for(auto it = candidate_set.begin(); it != candidate_set.end(); it++){
//            if(mmr_dist < it->first){
//                mmr_dist = it->first;
//                index = it;
//            }
//        }
//        char *element_mmr = getDataByInternalId(index->second);
//        dist_t element_mmr_dist = fstdistfunc_(data_point, element_mmr, dist_func_param_);
//        std::pair<dist_t, tableint> e_insert = std::make_pair(element_mmr_dist,index->second);
////        向eps_for_diversity_search添加新点
//        insertEpsForDiversitySearch(eps_for_diversity_search,e_insert);
////        更新candidate_set
////        更新已有邻居
//        candidate_set.erase(index);
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
//    }

};
}  // namespace D-hnswlib
