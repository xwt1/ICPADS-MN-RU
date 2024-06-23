#cmake_minimum_required(VERSION 3.22)
#project(graph-search VERSION 1.0)
#
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
##SET(CMAKE_C_COMPILER "/usr/bin/gcc")
##SET(CMAKE_CXX_COMPILER "/usr/bin/g++")
#
## 定义一个宏，将项目根目录的路径传递给C++代码
#add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
#
## 查找第三方库
#
## 添加原生的HNSW
#add_subdirectory(external/hnswlib)
#set_target_properties(hnswlib PROPERTIES OUTPUT_NAME original_hnswlib)
#
## 添加更改后的HNSW
#add_subdirectory(external/hnswlib_delete/hnswlib)
#set_target_properties(hnswlib PROPERTIES OUTPUT_NAME modified_hnswlib)
#
#
## 添加子目录，这会创建delete_update的目标
#add_subdirectory(src)
#
#
## 链接HNSWlib库到您的应用程序
#
#target_link_libraries(delete_update_lose_recall PRIVATE modified_hnswlib)

cmake_minimum_required(VERSION 3.22)
project(graph-search VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义一个宏，将项目根目录的路径传递给C++代码
add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

# 添加原生的 HNSW 库，保持默认目标名称 hnswlib
add_subdirectory(external/hnswlib)

# 添加更改后的 HNSW 库，设置 HNSWLIB_TARGET 变量
set(HNSWLIB_TARGET maintenance_hnswlib)
add_subdirectory(external/hnswlib_delete/hnswlib)
unset(HNSWLIB_TARGET) # 确保变量不再被后续使用

set(HNSWLIB_TARGET7 edge_connect_replaced_update7_hnswlib)
add_subdirectory(external/hnswlib_method7)
unset(HNSWLIB_TARGET7)

set(HNSWLIB_TARGET8 edge_connect_replaced_update8_hnswlib)
add_subdirectory(external/hnswlib_method8)
unset(HNSWLIB_TARGET8)

set(HNSWLIB_TARGET9 edge_connect_replaced_update9_hnswlib)
add_subdirectory(external/hnswlib_method9)
unset(HNSWLIB_TARGET9)

set(HNSWLIB_TARGET10 edge_connect_replaced_update10_hnswlib)
add_subdirectory(external/hnswlib_method10)
unset(HNSWLIB_TARGET10)


# 添加子目录，这会创建 delete_update 的目标
add_subdirectory(src)

# 链接修改后的 HNSW 库到您的应用程序
#target_link_libraries(delete_update_lose_recall PRIVATE maintenance_hnswlib)
#target_link_libraries(compute_delete_update PRIVATE maintenance_hnswlib)
#target_link_libraries(freshdiskann_prove PRIVATE hnswlib)
#target_link_libraries(generate_freshdiskann_prove_index PRIVATE hnswlib)
#target_link_libraries(generate_freshdiskann_groundtruth PRIVATE hnswlib)
#target_link_libraries(direct_delete_prove PRIVATE maintenance_hnswlib)
#target_link_libraries(mark_delete PRIVATE hnswlib)
#target_link_libraries(direct_delete_prove_mult PRIVATE maintenance_hnswlib)
#target_link_libraries(mark_delete_mult PRIVATE hnswlib)
#target_link_libraries(generate_direct_index PRIVATE maintenance_hnswlib)






# compare the qurey and insert and prove unreachable_points_phenomenon
target_link_libraries(compare_search_update_in_different_dataset PRIVATE hnswlib)
target_link_libraries(compute_imagenet_groundtruth PRIVATE hnswlib)
target_link_libraries(unreachable_points_phenomenon PRIVATE hnswlib)
target_link_libraries(unreachable_points_phenomenon_extream PRIVATE hnswlib)

# compare direct_delete and mark_delete with mult data
target_link_libraries(direct_delete_prove_mult_sift PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_mult_sift PRIVATE hnswlib)
target_link_libraries(direct_delete_prove_mult_gist PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_mult_gist PRIVATE hnswlib)
target_link_libraries(compute_gist_1M_groundTruth PRIVATE hnswlib)
target_link_libraries(direct_delete_prove_mult_imageNet PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_mult_imageNet PRIVATE hnswlib)

# compare direct_delete and mark_delete with random data
target_link_libraries(direct_delete_prove_single_sift PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_single_sift PRIVATE hnswlib)
target_link_libraries(direct_delete_prove_single_gist PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_single_gist PRIVATE hnswlib)
target_link_libraries(direct_delete_prove_single_imageNet PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_single_imageNet PRIVATE hnswlib)

# compare direct_delete and mark_delete with insert new data
target_link_libraries(direct_delete_prove_insert_new_sift PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_insert_new_sift PRIVATE hnswlib)
target_link_libraries(compute_sift_groundtruth PRIVATE hnswlib)

# test
target_link_libraries(test_query_direct_gist PRIVATE maintenance_hnswlib)
target_link_libraries(test_query_mark_gist PRIVATE hnswlib)






#compute_ground_truth
target_link_libraries(compute_netflix_groundtruth PRIVATE hnswlib)

# generate the index
target_link_libraries(hnsw_maintenance_generate_sift_index PRIVATE maintenance_hnswlib)
target_link_libraries(hnsw_prime_generate_sift_index PRIVATE hnswlib)
target_link_libraries(hnsw_maintenance_generate_gist_index PRIVATE maintenance_hnswlib)
target_link_libraries(hnsw_prime_generate_gist_index PRIVATE hnswlib)
target_link_libraries(hnsw_maintenance_generate_imageNet_index PRIVATE maintenance_hnswlib)
target_link_libraries(hnsw_prime_generate_imageNet_index PRIVATE hnswlib)
target_link_libraries(hnsw_maintenance_generate_sift2M_index PRIVATE maintenance_hnswlib)
target_link_libraries(hnsw_prime_generate_sift2M_index PRIVATE hnswlib)
target_link_libraries(hnsw_prime_generate_netflix_index PRIVATE hnswlib)

# generate random indice
target_link_libraries(generate_gist_random_data PRIVATE hnswlib)
target_link_libraries(generate_imageNet_random_data PRIVATE hnswlib)
target_link_libraries(generate_sift_random_data PRIVATE hnswlib)


# full_coverage:
# Edge-Connected replaced_update
target_link_libraries(edge_connected_replaced_update_7_full_coverage_sift PRIVATE edge_connect_replaced_update7_hnswlib)
target_link_libraries(edge_connected_replaced_update_8_full_coverage_sift PRIVATE edge_connect_replaced_update8_hnswlib)
target_link_libraries(edge_connected_replaced_update_9_full_coverage_sift PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(edge_connected_replaced_update_10_full_coverage_sift PRIVATE edge_connect_replaced_update10_hnswlib)
target_link_libraries(replaced_update_full_coverage_sift PRIVATE hnswlib)

target_link_libraries(edge_connected_replaced_update_7_full_coverage_gist PRIVATE edge_connect_replaced_update7_hnswlib)
target_link_libraries(edge_connected_replaced_update_8_full_coverage_gist PRIVATE edge_connect_replaced_update8_hnswlib)
target_link_libraries(edge_connected_replaced_update_9_full_coverage_gist PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(edge_connected_replaced_update_10_full_coverage_gist PRIVATE edge_connect_replaced_update10_hnswlib)
target_link_libraries(replaced_update_full_coverage_gist PRIVATE hnswlib)

target_link_libraries(edge_connected_replaced_update_7_full_coverage_imageNet PRIVATE edge_connect_replaced_update7_hnswlib)
target_link_libraries(edge_connected_replaced_update_8_full_coverage_imageNet PRIVATE edge_connect_replaced_update8_hnswlib)
target_link_libraries(edge_connected_replaced_update_9_full_coverage_imageNet PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(edge_connected_replaced_update_10_full_coverage_imageNet PRIVATE edge_connect_replaced_update10_hnswlib)
target_link_libraries(replaced_update_full_coverage_imageNet PRIVATE hnswlib)

# random
# Edge-Connected replaced_update
target_link_libraries(edge_connected_replaced_update_7_random_sift PRIVATE edge_connect_replaced_update7_hnswlib)
target_link_libraries(edge_connected_replaced_update_8_random_sift PRIVATE edge_connect_replaced_update8_hnswlib)
target_link_libraries(edge_connected_replaced_update_9_random_sift PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(edge_connected_replaced_update_10_random_sift PRIVATE edge_connect_replaced_update10_hnswlib)
target_link_libraries(replaced_update_random_sift PRIVATE hnswlib)

target_link_libraries(edge_connected_replaced_update_7_random_gist PRIVATE edge_connect_replaced_update7_hnswlib)
target_link_libraries(edge_connected_replaced_update_8_random_gist PRIVATE edge_connect_replaced_update8_hnswlib)
target_link_libraries(edge_connected_replaced_update_9_random_gist PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(edge_connected_replaced_update_10_random_gist PRIVATE edge_connect_replaced_update10_hnswlib)
target_link_libraries(replaced_update_random_gist PRIVATE hnswlib)

target_link_libraries(edge_connected_replaced_update_7_random_imageNet PRIVATE edge_connect_replaced_update7_hnswlib)
target_link_libraries(edge_connected_replaced_update_8_random_imageNet PRIVATE edge_connect_replaced_update8_hnswlib)
target_link_libraries(edge_connected_replaced_update_9_random_imageNet PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(edge_connected_replaced_update_10_random_imageNet PRIVATE edge_connect_replaced_update10_hnswlib)
target_link_libraries(replaced_update_random_imageNet PRIVATE hnswlib)



target_link_libraries(edge_connected_replaced_update_9_full_coverage_netflix PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(replaced_update_full_coverage_netflix PRIVATE hnswlib)



# new insert
target_link_libraries(edge_connected_replaced_update_7_new_insert_sift_2M PRIVATE edge_connect_replaced_update7_hnswlib)
target_link_libraries(edge_connected_replaced_update_8_new_insert_sift_2M PRIVATE edge_connect_replaced_update8_hnswlib)
target_link_libraries(edge_connected_replaced_update_9_new_insert_sift_2M PRIVATE edge_connect_replaced_update9_hnswlib)
target_link_libraries(edge_connected_replaced_update_10_new_insert_sift_2M PRIVATE edge_connect_replaced_update10_hnswlib)
target_link_libraries(replaced_update_new_insert_sift_2M PRIVATE hnswlib)

# count end recall
# notice that the search process is just the same, so we use the prime hnswlib
target_link_libraries(end_recall_full_coverage PRIVATE hnswlib)
target_link_libraries(end_recall_random PRIVATE hnswlib)
target_link_libraries(end_recall_new_insert PRIVATE hnswlib)