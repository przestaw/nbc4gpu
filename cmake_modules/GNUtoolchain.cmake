find_program(GCC_AR gcc-ar)
if (GCC_AR)
    set(CMAKE_AR ${GCC_AR})
endif ()
find_program(GCC_RANLIB gcc-ranlib)
if (GCC_RANLIB)
    set(CMAKE_RANLIB ${GCC_RANLIB})
endif ()