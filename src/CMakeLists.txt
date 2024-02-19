## 定义一个函数来递归地收集所有头文件目录
#function(include_headers_recursively root_dir)
#    file(GLOB_RECURSE headers ${root_dir}/*.h)
#    set(dir_list "")
#    foreach(header_file ${headers})
#        get_filename_component(dir_path ${header_file} PATH)
#        set(dir_list ${dir_list} ${dir_path})
#    endforeach()
#    list(REMOVE_DUPLICATES dir_list)
#    include_directories(${dir_list})
#endfunction()
#
#include_headers_recursively(${PROJECT_SOURCE_DIR}/include)

# 包括子目录，这会执行子目录的CMakeLists.txt并更新SOURCE_FILES和HEADER_FILES变量
add_subdirectory(construction)
add_subdirectory(data-structure)

message(STATUS "shit ${CONSTRUCTION_HEADERS}")
message(STATUS "wtf1231 ${DATA_STRUCTURE_HEADERS}")
#
#message(status "${PROJECT_SOURCE_DIR}/include")
#message(status "${CONSTRUCTION_HEADERS}")
#message(status "${DATA_STRUCTURE_HEADERS}")

# 包含头文件目录
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${CONSTRUCTION_HEADERS}
        ${DATA_STRUCTURE_HEADERS}
)

# 添加源文件
set(SOURCE_FILES main.cpp)

# 附加来自子目录的源文件
list(APPEND SOURCE_FILES ${CONSTRUCTION_SOURCES})
list(APPEND SOURCE_FILES ${DATA_STRUCTURE_SOURCES})

message(STATUS "shit23 ${CONSTRUCTION_SOURCES}")
message(STATUS "wtf123145 ${DATA_STRUCTURE_SOURCES}")

# 添加可执行文件，包括源文件和头文件
add_executable(graph-search ${SOURCE_FILES})
