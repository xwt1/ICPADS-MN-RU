#file(GLOB DATA_STRUCTURE_SOURCES "*.cpp")
##file(GLOB DATA_STRUCTURE_HEADERS "*.h")
#
#
#set(DATA_STRUCTURE_SOURCES ${DATA_STRUCTURE_SOURCES} PARENT_SCOPE)
##set(DATA_STRUCTURE_HEADERS ${DATA_STRUCTURE_HEADERS} PARENT_SCOPE)

set(ISSUE_STATEMENT_HEADERS
        ${PROJECT_SOURCE_DIR}/include/issue_statement
        PARENT_SCOPE
        )

set(delete_update_lose_recall
        ${CMAKE_CURRENT_SOURCE_DIR}/delete_update_lose_recall.cpp
        PARENT_SCOPE
        )