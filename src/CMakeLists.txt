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





#compute_ground_truth
add_executable(compute_netflix_groundtruth compute_groundtruth/compute_netflix_groundtruth.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(compute_imageNet2M_groundtruth compute_groundtruth/compute_imageNet_2M_groundTruth.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(compute_word2vec_groundtruth compute_groundtruth/compute_groundtruth_word2vec.cpp
        util/util.cpp
        util/thread_pool.cpp)

# generate the index
add_executable(hnsw_maintenance_generate_sift_index generate_index/hnsw_maintenance/generate_sift_index.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(hnsw_prime_generate_sift_index generate_index/hnsw_prime/generate_sift_index.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(hnsw_maintenance_generate_gist_index generate_index/hnsw_maintenance/generate_gist_index.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(hnsw_prime_generate_gist_index generate_index/hnsw_prime/generate_gist_index.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(hnsw_maintenance_generate_imageNet_index generate_index/hnsw_maintenance/generate_imageNet_index.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(hnsw_maintenance_generate_imageNet_index_2 generate_index/hnsw_maintenance/generate_imageNet_index_test.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(hnsw_prime_generate_imageNet_index generate_index/hnsw_prime/generate_imageNet_index.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(hnsw_maintenance_generate_sift2M_index generate_index/hnsw_maintenance/generate_sift2M_index.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(hnsw_prime_generate_sift2M_index generate_index/hnsw_prime/generate_sift2M_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(hnsw_prime_generate_netflix_index generate_index/hnsw_prime/generate_netflix_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(hnsw_maintenance_generate_msong_index generate_index/hnsw_maintenance/generate_msong_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(hnsw_maintenance_generate_word_vec_index generate_index/hnsw_maintenance/generate_word_vec_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(faiss_generate_index_gist generate_index/faiss/gist/generate_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(faiss_generate_index_sift generate_index/faiss/sift/generate_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(faiss_generate_index_imageNet generate_index/faiss/imageNet/generate_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(faiss_generate_index_sift2M generate_index/faiss/sift2M/generate_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(faiss_generate_index_word2vec generate_index/faiss/word2vec/generate_index.cpp
        util/util.cpp
        util/thread_pool.cpp)

# generate random indice
add_executable(generate_gist_random_data replaced_update_vaild/compare_edge_connect_with_prime_random/gist/generate_random.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(generate_imageNet_random_data replaced_update_vaild/compare_edge_connect_with_prime_random/imageNet/generate_random.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(generate_sift_random_data replaced_update_vaild/compare_edge_connect_with_prime_random/sift/generate_random.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(generate_sift100M_random_data replaced_update_vaild/Sift100M/random/generate_random.cpp
        util/util.cpp
        util/thread_pool.cpp)


# full_coverage:
# Edge-Connected replaced_update
add_executable(edge_connected_replaced_update_7_full_coverage_sift replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/sift/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_full_coverage_sift replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/sift/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_full_coverage_sift replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/sift/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_full_coverage_sift replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/sift/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_full_coverage_sift replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/sift/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(edge_connected_replaced_update_7_full_coverage_gist replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/gist/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_full_coverage_gist replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/gist/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_full_coverage_gist replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/gist/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_full_coverage_gist replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/gist/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_full_coverage_gist replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/gist/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(faiss_full_coverage_gist replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/gist/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_full_coverage_imageNet replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_full_coverage_sift replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/sift/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_full_coverage_word2vec replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)


add_executable(edge_connected_replaced_update_7_full_coverage_word2vec replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_full_coverage_word2vec replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_full_coverage_word2vec replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_full_coverage_word2vec replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_full_coverage_word2vec replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)


add_executable(edge_connected_replaced_update_7_full_coverage_imageNet replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_full_coverage_imageNet replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_full_coverage_imageNet replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_full_coverage_imageNet replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_full_coverage_imageNet replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)


add_executable(edge_connected_replaced_update_7_full_coverage_imageNet2 replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet2/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_full_coverage_imageNet2 replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet2/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_full_coverage_imageNet2 replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet2/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_full_coverage_imageNet2 replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet2/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_full_coverage_imageNet2 replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet2/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)



add_executable(replaced_update_full_coverage_netflix replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/netflix/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(edge_connected_replaced_update_9_full_coverage_netflix replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/netflix/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)

# random
# Edge-Connected replaced_update
add_executable(edge_connected_replaced_update_7_random_sift replaced_update_vaild/compare_edge_connect_with_prime_random/sift/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_random_sift replaced_update_vaild/compare_edge_connect_with_prime_random/sift/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_random_sift replaced_update_vaild/compare_edge_connect_with_prime_random/sift/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_random_sift replaced_update_vaild/compare_edge_connect_with_prime_random/sift/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_random_sift replaced_update_vaild/compare_edge_connect_with_prime_random/sift/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_random_sift replaced_update_vaild/compare_edge_connect_with_prime_random/sift/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)


add_executable(edge_connected_replaced_update_7_random_gist replaced_update_vaild/compare_edge_connect_with_prime_random/gist/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_random_gist replaced_update_vaild/compare_edge_connect_with_prime_random/gist/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_random_gist replaced_update_vaild/compare_edge_connect_with_prime_random/gist/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_random_gist replaced_update_vaild/compare_edge_connect_with_prime_random/gist/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_random_gist replaced_update_vaild/compare_edge_connect_with_prime_random/gist/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_random_gist replaced_update_vaild/compare_edge_connect_with_prime_random/gist/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(edge_connected_replaced_update_7_random_imageNet replaced_update_vaild/compare_edge_connect_with_prime_random/imageNet/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_random_imageNet replaced_update_vaild/compare_edge_connect_with_prime_random/imageNet/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_random_imageNet replaced_update_vaild/compare_edge_connect_with_prime_random/imageNet/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_random_imageNet replaced_update_vaild/compare_edge_connect_with_prime_random/imageNet/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_random_imageNet replaced_update_vaild/compare_edge_connect_with_prime_random/imageNet/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_random_imageNet replaced_update_vaild/compare_edge_connect_with_prime_random/imageNet/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)

# insert new data
add_executable(edge_connected_replaced_update_7_new_insert_sift_2M replaced_update_vaild/compare_edge_connect_with_prime_new_insert/sift_2M/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_new_insert_sift_2M replaced_update_vaild/compare_edge_connect_with_prime_new_insert/sift_2M/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_new_insert_sift_2M replaced_update_vaild/compare_edge_connect_with_prime_new_insert/sift_2M/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_new_insert_sift_2M replaced_update_vaild/compare_edge_connect_with_prime_new_insert/sift_2M/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_new_insert_sift_2M replaced_update_vaild/compare_edge_connect_with_prime_new_insert/sift_2M/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_full_coverage_sift2M replaced_update_vaild/compare_edge_connect_with_prime_new_insert/sift_2M/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)


# count end recall
add_executable(end_recall_full_coverage replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/end_recall.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_random replaced_update_vaild/compare_edge_connect_with_prime_random/end_recall.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_new_insert replaced_update_vaild/compare_edge_connect_with_prime_new_insert/end_recall.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_gist_full_coverage replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/end_recall_gist.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_gist_random replaced_update_vaild/compare_edge_connect_with_prime_random/end_recall_gist.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_imageNet_2M_full_coverage replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/end_recall_imageNet2M.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_imageNet_2M_random replaced_update_vaild/compare_edge_connect_with_prime_random/end_recall_imageNet2M.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_faiss_gist replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/gist/end_recall_faiss_gist.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_faiss_imageNet2M replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet/end_recall_faiss_imageNet2M.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_faiss_sift replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/sift/end_recall_faiss_sift.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_faiss_word2vec_full_coverage replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/end_recall_faiss_word2vec.cpp
        util/util.cpp
        util/thread_pool.cpp)

add_executable(end_recall_faiss_gist_random replaced_update_vaild/compare_edge_connect_with_prime_random/end_recall_faiss_gist.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_faiss_imageNet_random replaced_update_vaild/compare_edge_connect_with_prime_random/end_recall_faiss_imageNet.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_faiss_sift_random replaced_update_vaild/compare_edge_connect_with_prime_random/end_recall_faiss_sift.cpp
        util/util.cpp
        util/thread_pool.cpp)



# count end recall
add_executable(end_recall_full_coverage_imageNet2 replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/imageNet2/end_recall.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(end_recall_full_coverage_word2vec replaced_update_vaild/compare_edge_connect_with_prime_full_coverage/word2vec/end_recall_word2vec.cpp
        util/util.cpp
        util/thread_pool.cpp)


# compare the qurey and insert and prove unreachable_points_phenomenon
add_executable(compare_search_update_in_different_dataset hnsw_prime_search_mark_delete_compare/compare_search_update_in_different_dataset.cpp
                                                    util/util.cpp
                                                    util/thread_pool.cpp)
add_executable(compute_imagenet_groundtruth compute_groundtruth/compute_imagenet_groundtruth.cpp
                                                    util/util.cpp
        util/thread_pool.cpp)
add_executable(unreachable_points_phenomenon unreachable_points_phenomenon/unreachable_points_phenomenon.cpp
                                                    util/util.cpp
        util/thread_pool.cpp)
add_executable(unreachable_points_phenomenon_extream unreachable_points_phenomenon/unreachable_points_phenomenon_extream.cpp
                                                    util/util.cpp
        util/thread_pool.cpp)

# compare direct_delete and mark_delete with mult data
add_executable(direct_delete_prove_mult_sift direct_delete/compare_direct_delete_and_mark_mult_batch/sift/direct_delete_prove.cpp
                                                    util/direct_delete_util.cpp
        util/thread_pool.cpp)
add_executable(mark_delete_mult_sift direct_delete/compare_direct_delete_and_mark_mult_batch/sift/mark_delete.cpp
                                                    util/util.cpp
        util/thread_pool.cpp)
add_executable(direct_delete_prove_mult_gist direct_delete/compare_direct_delete_and_mark_mult_batch/gist/direct_delete_prove.cpp
                                                    util/direct_delete_util.cpp
        util/thread_pool.cpp)
add_executable(mark_delete_mult_gist direct_delete/compare_direct_delete_and_mark_mult_batch/gist/mark_delete.cpp
                                                    util/util.cpp
        util/thread_pool.cpp)
add_executable(compute_gist_1M_groundTruth compute_groundtruth/compute_gist_1M_groundTruth.cpp
                                                    util/util.cpp
        util/thread_pool.cpp)

add_executable(direct_delete_prove_mult_imageNet direct_delete/compare_direct_delete_and_mark_mult_batch/imageNet/direct_delete_prove.cpp
                                                    util/direct_delete_util.cpp
        util/thread_pool.cpp)
add_executable(mark_delete_mult_imageNet direct_delete/compare_direct_delete_and_mark_mult_batch/imageNet/mark_delete.cpp
                                                    util/util.cpp
        util/thread_pool.cpp)

# compare direct_delete and mark_delete with random data
add_executable(direct_delete_prove_single_sift direct_delete/compare_direct_delete_and_mark_single_batch/sift/direct_delete_prove.cpp
                                                util/direct_delete_util.cpp
        util/thread_pool.cpp)
add_executable(mark_delete_single_sift direct_delete/compare_direct_delete_and_mark_single_batch/sift/mark_delete.cpp
                                                util/util.cpp
        util/thread_pool.cpp)
add_executable(direct_delete_prove_single_gist direct_delete/compare_direct_delete_and_mark_single_batch/gist/direct_delete_prove.cpp
                                                util/direct_delete_util.cpp
        util/thread_pool.cpp)
add_executable(mark_delete_single_gist direct_delete/compare_direct_delete_and_mark_single_batch/gist/mark_delete.cpp
                                                util/util.cpp
        util/thread_pool.cpp)
add_executable(direct_delete_prove_single_imageNet direct_delete/compare_direct_delete_and_mark_single_batch/imageNet/direct_delete_prove.cpp
                                                util/direct_delete_util.cpp
        util/thread_pool.cpp)
add_executable(mark_delete_single_imageNet direct_delete/compare_direct_delete_and_mark_single_batch/imageNet/mark_delete.cpp
                                                util/util.cpp
        util/thread_pool.cpp)

# compare direct_delete and mark_delete with insert new data
add_executable(direct_delete_prove_insert_new_sift direct_delete/compare_direct_delete_and_mark_add_new_point/sift/direct_delete_prove.cpp
                                                util/direct_delete_util.cpp
        util/thread_pool.cpp)
add_executable(mark_delete_insert_new_sift direct_delete/compare_direct_delete_and_mark_add_new_point/sift/mark_delete.cpp
                                                util/util.cpp
        util/thread_pool.cpp)
add_executable(compute_sift_groundtruth compute_groundtruth/compute_sift_groundtruth.cpp
                                                util/util.cpp
        util/thread_pool.cpp)

# Sift100M
add_executable(edge_connected_replaced_update_7_full_coverage_sift100M replaced_update_vaild/Sift100M/full_coverage/edge_connected_7.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_8_full_coverage_sift100M replaced_update_vaild/Sift100M/full_coverage/edge_connected_8.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_9_full_coverage_sift100M replaced_update_vaild/Sift100M/full_coverage/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(edge_connected_replaced_update_10_full_coverage_sift100M replaced_update_vaild/Sift100M/full_coverage/edge_connected_10.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(replaced_update_full_coverage_sift100M replaced_update_vaild/Sift100M/full_coverage/replaced_update.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(faiss_full_coverage_sift100M replaced_update_vaild/Sift100M/full_coverage/faiss_ivf_flat.cpp
        util/util.cpp
        util/thread_pool.cpp)




# Back Up Index Build
add_executable(back_up_edge_connected_9 replaced_update_vaild/back_up_index_compare/gist/edge_connected_9.cpp
        util/util.cpp
        util/thread_pool.cpp)

# test
add_executable(test_query_direct_gist test/C++/test_query_direct_gist.cpp
                                                util/util.cpp
        util/thread_pool.cpp)
add_executable(test_query_mark_gist test/C++/test_query_mark_gist.cpp
                                                util/util.cpp
        util/thread_pool.cpp)
add_executable(test_degree_percentage_sift test/C++/test_degree_percentage_sift.cpp
                                                util/util.cpp
        util/thread_pool.cpp)
add_executable(test_degree_percentage_gist test/C++/test_degree_percentage_gist.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(test_degree_percentage_imageNet test/C++/test_degree_percentage_imageNet.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(test_degree_percentage_msong test/C++/test_degree_percentage_msong.cpp
        util/util.cpp
        util/thread_pool.cpp)
add_executable(test_degree_percentage_word_vec test/C++/test_degree_percentage_word_vec.cpp
        util/util.cpp
        util/thread_pool.cpp)

#add_executable(delete_update_lose_recall issue_statement/delete_update_lose_recall.cpp)
#add_executable(compute_delete_update issue_statement/compute_delete_update.cpp)
#add_executable(freshdiskann_prove freshdiskann_prove/freshdiskann_prove.cpp)
#add_executable(generate_freshdiskann_prove_index freshdiskann_prove/generate_freshdiskann_prove_index.cpp)
#add_executable(generate_freshdiskann_groundtruth freshdiskann_prove/generate_freshdiskann_groundtruth.cpp)
#
#add_executable(direct_delete_prove_mult direct_delete/compare_direct_delete_and_mark_mult_batch/direct_delete_prove.cpp)
#add_executable(mark_delete_mult direct_delete/compare_direct_delete_and_mark_mult_batch/mark_delete.cpp)
#
#add_executable(direct_delete_prove direct_delete/compare_direct_delete_and_mark_single_batch/direct_delete_prove.cpp)
#add_executable(mark_delete direct_delete/compare_direct_delete_and_mark_single_batch/mark_delete.cpp)
#add_executable(generate_direct_index direct_delete/generate_index/generate_direct_index.cpp)

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