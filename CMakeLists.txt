cmake_minimum_required(VERSION 3.22)
project(graph-search VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#SET(CMAKE_C_COMPILER "/usr/bin/gcc")
#SET(CMAKE_CXX_COMPILER "/usr/bin/g++")

# 定义一个宏，将项目根目录的路径传递给C++代码
add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

# 查找第三方库

# 查找OpenMP
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found.")
endif()

# 查找MLPACK
find_package(MLPACK)
if(NOT MLPACK_FOUND)
    message(FATAL_ERROR "MLPACK not found. Please run: apt-get install libmlpack-dev")
endif()

# 找到Eigen库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please run: sudo apt-get install libeigen3-dev")
endif()

# 包括HNSWlib目录与NN-Decent
#add_subdirectory(external/hnswlib)
add_subdirectory(external/D-hnswlib)

include_directories(external/nndescent/include)

# 添加子目录，这会创建graph-search目标
add_subdirectory(src)

# 现在，我们可以链接第三方库到graph-search目标
if(OpenMP_CXX_FOUND)
    # 添加OpenMP编译和链接选项到graph-search目标
    target_compile_options(graph-search PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(graph-search PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(graph-search PRIVATE ${OpenMP_CXX_FLAGS})

    # 添加OpenMP编译和链接选项到graph-search目标
    target_compile_options(compare_rawMMR_With_Dhnsw PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(compare_rawMMR_With_Dhnsw PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(compare_rawMMR_With_Dhnsw PRIVATE ${OpenMP_CXX_FLAGS})

    # 添加OpenMP编译和链接选项到graph-search目标
    target_compile_options(generate_index PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(generate_index PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(generate_index PRIVATE ${OpenMP_CXX_FLAGS})

    # 添加OpenMP编译和链接选项到graph-search目标
    target_compile_options(compare_rawMMR_with_Dhnsw_in_ip PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(compare_rawMMR_with_Dhnsw_in_ip PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(compare_rawMMR_with_Dhnsw_in_ip PRIVATE ${OpenMP_CXX_FLAGS})

    target_compile_options(generate_index_ip PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(generate_index_ip PRIVATE OpenMP::OpenMP_CXX)
    target_link_options(generate_index_ip PRIVATE ${OpenMP_CXX_FLAGS})

endif()

# 链接MLPACK库到graph-search目标
target_link_libraries(graph-search PRIVATE mlpack)
target_link_libraries(compare_rawMMR_With_Dhnsw PRIVATE mlpack)
target_link_libraries(generate_index PRIVATE mlpack)
target_link_libraries(compare_rawMMR_with_Dhnsw_in_ip PRIVATE mlpack)
target_link_libraries(generate_index_ip PRIVATE mlpack)

# 链接HNSWlib库到您的应用程序
target_link_libraries(graph-search PRIVATE hnswlib)
target_link_libraries(compare_rawMMR_With_Dhnsw PRIVATE hnswlib)
target_link_libraries(generate_index PRIVATE hnswlib)
target_link_libraries(compare_rawMMR_with_Dhnsw_in_ip PRIVATE hnswlib)
target_link_libraries(generate_index_ip PRIVATE hnswlib)

# 链接Eigen库
target_link_libraries(compare_rawMMR_With_Dhnsw PRIVATE Eigen3::Eigen)
target_link_libraries(compare_rawMMR_with_Dhnsw_in_ip PRIVATE Eigen3::Eigen)
target_link_libraries(graph-search PRIVATE Eigen3::Eigen)
target_link_libraries(generate_index PRIVATE Eigen3::Eigen)
target_link_libraries(generate_index_ip PRIVATE Eigen3::Eigen)