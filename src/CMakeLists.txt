# 包括子目录，这会执行子目录的CMakeLists.txt并更新SOURCE_FILES和HEADER_FILES变量
#add_subdirectory(construction)
#add_subdirectory(data-structure)
#add_subdirectory(file)
#add_subdirectory(evaluate)
#add_subdirectory(issue_statement)

#add_subdirectory(util)


#!!!记得添加子目录

#message(STATUS "shit ${CONSTRUCTION_HEADERS}")
#message(STATUS "wtf1231 ${DATA_STRUCTURE_HEADERS}")


# 包含头文件目录
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/util
#        ${CONSTRUCTION_HEADERS}
#        ${DATA_STRUCTURE_HEADERS}
#        ${FILE_HEADERS}
#        ${EVALUATE_HEADERS}
        #如果有新的目录,模仿前面的句子在这里添加新的头文件目录
#        ${ISSUE_STATEMENT_HEADERS}
#        ${UTIL_HEADER}
)




# 添加源文件
add_executable(hnsw_maintenance_generate_sift_index generate_index/hnsw_maintenance/generate_sift_index.cpp
                                                    util/util.cpp)
add_executable(hnsw_prime_generate_sift_index generate_index/hnsw_prime/generate_sift_index.cpp
                                                    util/util.cpp)
add_executable(hnsw_maintenance_generate_gist_index generate_index/hnsw_maintenance/generate_gist_index.cpp
        util/util.cpp)
add_executable(hnsw_prime_generate_gist_index generate_index/hnsw_prime/generate_gist_index.cpp
        util/util.cpp)
add_executable(hnsw_maintenance_generate_imageNet_index generate_index/hnsw_maintenance/generate_imageNet_index.cpp
        util/util.cpp)
add_executable(hnsw_prime_generate_imageNet_index generate_index/hnsw_prime/generate_imageNet_index.cpp
        util/util.cpp)

add_executable(compare_search_update_in_different_dataset hnsw_prime_search_mark_delete_compare/compare_search_update_in_different_dataset.cpp
                                                    util/util.cpp)
add_executable(compute_imagenet_groundtruth.cpp compute_groundtruth/compute_imagenet_groundtruth.cpp
                                                    util/util.cpp)





add_executable(delete_update_lose_recall issue_statement/delete_update_lose_recall.cpp)
add_executable(compute_delete_update issue_statement/compute_delete_update.cpp)
add_executable(freshdiskann_prove freshdiskann_prove/freshdiskann_prove.cpp)
add_executable(generate_freshdiskann_prove_index freshdiskann_prove/generate_freshdiskann_prove_index.cpp)
add_executable(generate_freshdiskann_groundtruth freshdiskann_prove/generate_freshdiskann_groundtruth.cpp)

add_executable(direct_delete_prove_mult direct_delete/compare_direct_delete_and_mark_mult_batch/direct_delete_prove.cpp)
add_executable(mark_delete_mult direct_delete/compare_direct_delete_and_mark_mult_batch/mark_delete.cpp)

add_executable(direct_delete_prove direct_delete/compare_direct_delete_and_mark_single_batch/direct_delete_prove.cpp)
add_executable(mark_delete direct_delete/compare_direct_delete_and_mark_single_batch/mark_delete.cpp)
add_executable(generate_direct_index direct_delete/generate_index/generate_direct_index.cpp)

##set(SOURCE_FILES main.cpp)
#set(SOURCE_FILES)
#
## 附加来自子目录的源文件
##list(APPEND SOURCE_FILES ${CONSTRUCTION_SOURCES})
##list(APPEND SOURCE_FILES ${DATA_STRUCTURE_SOURCES})
##list(APPEND SOURCE_FILES ${FILE_SOURCES})
##list(APPEND SOURCE_FILES ${EVALUATE_SOURCES})
#list(APPEND SOURCE_FILES ${EVALUATE_SOURCES})
#
##如果有新的源文件目录,模仿前面的句子在这里添加新的源文件目录
#
#
## 添加可执行文件，包括源文件和头文件
#add_executable(graph-search ${SOURCE_FILES} main.cpp)
#add_executable(compare_rawMMR_With_Dhnsw ${SOURCE_FILES} ./test/compare_rawMMR_With_Dhnsw.cpp)
#add_executable(compare_rawMMR_with_Dhnsw_in_ip ${SOURCE_FILES} ./test/compare_rawMMR_with_Dhnsw_in_ip.cpp)
#add_executable(generate_index ${SOURCE_FILES} ./test/generate_index.cpp)
#add_executable(generate_index_ip ${SOURCE_FILES} ./test/generate_index_ip.cpp)
#
#add_executable(generate_mmr_index ./test/generate_mmr_index.cpp)
#add_executable(delete_update_connection ./test/delete_update.cpp)
#add_executable(tes_mmr_and_normal ./test/tes_mmr_and_normal.cpp)
#add_executable(export_cluster_graph ./test/export_cluster_graph.cpp)
#add_executable(test_new_level_selection test/test_new_level_selection/test_new_level_selection.cpp)
#add_executable(test_new_level_selection2 test/test_new_level_selection/test_new_level_selection2.cpp)
#add_executable(search_performance test/test_new_level_selection/search_performance.cpp)