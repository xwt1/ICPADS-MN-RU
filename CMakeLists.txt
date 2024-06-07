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

# 添加子目录，这会创建 delete_update 的目标
add_subdirectory(src)

# 链接修改后的 HNSW 库到您的应用程序
target_link_libraries(delete_update_lose_recall PRIVATE maintenance_hnswlib)
target_link_libraries(compute_delete_update PRIVATE maintenance_hnswlib)
target_link_libraries(freshdiskann_prove PRIVATE hnswlib)
target_link_libraries(generate_freshdiskann_prove_index PRIVATE hnswlib)
target_link_libraries(generate_freshdiskann_groundtruth PRIVATE hnswlib)
target_link_libraries(direct_delete_prove PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete PRIVATE hnswlib)
target_link_libraries(direct_delete_prove_mult PRIVATE maintenance_hnswlib)
target_link_libraries(mark_delete_mult PRIVATE hnswlib)
target_link_libraries(generate_direct_index PRIVATE maintenance_hnswlib)





target_link_libraries(hnsw_maintenance_generate_sift_index PRIVATE maintenance_hnswlib)
target_link_libraries(hnsw_prime_generate_sift_index PRIVATE hnswlib)
target_link_libraries(hnsw_maintenance_generate_gist_index PRIVATE maintenance_hnswlib)
target_link_libraries(hnsw_prime_generate_gist_index PRIVATE hnswlib)
target_link_libraries(hnsw_maintenance_generate_imageNet_index PRIVATE maintenance_hnswlib)
target_link_libraries(hnsw_prime_generate_imageNet_index PRIVATE hnswlib)
target_link_libraries(compare_search_update_in_different_dataset PRIVATE hnswlib)

target_link_libraries(compute_imagenet_groundtruth.cpp PRIVATE hnswlib)