cmake_minimum_required(VERSION 3.22)
project(graph-search VERSION 1.0)

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

# 添加子目录，这会创建graph-search目标
add_subdirectory(src)

# 现在，我们可以链接第三方库到graph-search目标
if(OpenMP_CXX_FOUND)
    # 添加OpenMP编译和链接选项到graph-search目标
    target_compile_options(graph-search PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(graph-search PRIVATE OpenMP::OpenMP_CXX)
endif()

# 链接MLPACK库到graph-search目标
target_link_libraries(graph-search PRIVATE mlpack)
