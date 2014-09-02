
SET(SOURCES "")
SET(HEADERS "")

FILE(GLOB SOURCES *.cpp)
FILE(GLOB HEADERS *.hpp)

add_library(albanySTKRebalance ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

