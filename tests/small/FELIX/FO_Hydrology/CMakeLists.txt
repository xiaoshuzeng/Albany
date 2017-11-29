# 1. Copy Input file from source to binary dir
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/input_fo_hydrology.xml
                ${CMAKE_CURRENT_BINARY_DIR}/input_fo_hydrology.xml COPYONLY)

GET_FILENAME_COMPONENT (testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#2. Tests
ADD_TEST (${testName}_fo_hydrology ${Albany.exe} input_fo_hydrology.xml)
