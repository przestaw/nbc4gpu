add_executable(${TESTCPU} ${LIB_INC} ${LIB_SRC} ${TEST_CPU_SRC} ${CPU_TMP})
target_link_libraries(${TESTCPU} ${EXTERNAL_LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test(runTest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TESTCPU}")
