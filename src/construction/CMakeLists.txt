add_subdirectory(util)

set(CONSTRUCTION_HEADERS
        ${PROJECT_SOURCE_DIR}/include/construction
        ${UTIL_HEADERS}
        PARENT_SCOPE
        )

set(CONSTRUCTION_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/base_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hierarchy_graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/node.cpp
        ${UTIL_SOURCES}
        PARENT_SCOPE
        )

