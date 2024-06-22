cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# 使用外部传入的变量设置目标名称
if(NOT DEFINED HNSWLIB_TARGET10)
    set(HNSWLIB_TARGET10 hnswlib)
endif()

add_library(${HNSWLIB_TARGET10} INTERFACE)
add_library(hnswlib::${HNSWLIB_TARGET10} ALIAS ${HNSWLIB_TARGET10})

target_include_directories(${HNSWLIB_TARGET10} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${HNSWLIB_TARGET10}
        EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
        FILE hnswlibConfig.cmake
        NAMESPACE hnswlib::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
        if(COMPILER_SUPPORT_NATIVE_FLAG)
            SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
            message("set -march=native flag")
        else()
            check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
            if(COMPILER_SUPPORT_M1_FLAG)
                SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
                message("set -mcpu=apple-m1 flag")
            endif()
        endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # examples
    add_executable(example_search examples/cpp/example_search.cpp)
    target_link_libraries(example_search ${HNSWLIB_TARGET10})

    add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    target_link_libraries(example_epsilon_search ${HNSWLIB_TARGET10})

    add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    target_link_libraries(example_multivector_search ${HNSWLIB_TARGET10})

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter ${HNSWLIB_TARGET10})

    add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    target_link_libraries(example_replace_deleted ${HNSWLIB_TARGET10})

    add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    target_link_libraries(example_mt_search ${HNSWLIB_TARGET10})

    add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    target_link_libraries(example_mt_filter ${HNSWLIB_TARGET10})

    add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    target_link_libraries(example_mt_replace_deleted ${HNSWLIB_TARGET10})

    # tests
    add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    target_link_libraries(multivector_search_test ${HNSWLIB_TARGET10})

    add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    target_link_libraries(epsilon_search_test ${HNSWLIB_TARGET10})

    add_executable(test_updates tests/cpp/updates_test.cpp)
    target_link_libraries(test_updates ${HNSWLIB_TARGET10})

    add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test ${HNSWLIB_TARGET10})

    add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    target_link_libraries(searchKnnWithFilter_test ${HNSWLIB_TARGET10})

    add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    target_link_libraries(multiThreadLoad_test ${HNSWLIB_TARGET10})

    add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    target_link_libraries(multiThread_replace_test ${HNSWLIB_TARGET10})

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_link_libraries(main ${HNSWLIB_TARGET10})
endif()
