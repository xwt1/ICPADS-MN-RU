# 包括子目录，这会执行子目录的CMakeLists.txt并更新SOURCE_FILES和HEADER_FILES变量
add_subdirectory(construction)
add_subdirectory(data-structure)
add_subdirectory(file)

#message(STATUS "shit ${CONSTRUCTION_HEADERS}")
#message(STATUS "wtf1231 ${DATA_STRUCTURE_HEADERS}")


# 包含头文件目录
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${CONSTRUCTION_HEADERS}
        ${DATA_STRUCTURE_HEADERS}
        ${FILE_HEADERS}
        #如果有新的目录,模仿前面的句子在这里添加新的头文件目录
)

# 添加源文件
set(SOURCE_FILES main.cpp)

# 附加来自子目录的源文件
list(APPEND SOURCE_FILES ${CONSTRUCTION_SOURCES})
list(APPEND SOURCE_FILES ${DATA_STRUCTURE_SOURCES})
list(APPEND SOURCE_FILES ${FILE_SOURCES})
#如果有新的源文件目录,模仿前面的句子在这里添加新的源文件目录


#message(STATUS "shit23 ${CONSTRUCTION_SOURCES}")
#message(STATUS "wtf123145 ${DATA_STRUCTURE_SOURCES}")

# 添加可执行文件，包括源文件和头文件
add_executable(graph-search ${SOURCE_FILES})
