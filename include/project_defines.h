#ifndef PROJECT_DEFINES_H
#define PROJECT_DEFINES_H

#define UNUSED_VAL(x) (void)(x)
#define CL_TARGET_OPENCL_VERSION 120

#if __GNUG__ || __clang__

#define DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wshadow\"") _Pragma("GCC diagnostic ignored \"-Wignored-qualifiers\"") // boost computes has a lot of warnings
#define DIAGNOSTIC_POP _Pragma("GCC diagnostic pop") // restore diagnostics

#else

#define DIAGNOSTIC_PUSH
#define DIAGNOSTIC_POP

#endif

#endif // PROJECT_DEFINES_H
