add_executable(${TESTGPU} ${LIB_INC} ${LIB_SRC} ${TEST_GPU_SRC} ${LIB_TMP})
target_link_libraries(${TESTGPU} ${EXTERNAL_LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test(runTest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TESTGPU}")
