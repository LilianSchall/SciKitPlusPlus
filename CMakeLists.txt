cmake_minimum_required(VERSION 3.22)

project(scikit++ LANGUAGES CXX)

find_package(benchmark)
find_package(GTest)
set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)

if (BLAS_FOUND)
    find_path(BLAS_INCLUDE_DIRS cblas.h
        /usr/include
        /usr/local/include
        $ENV{BLAS_HOME}/include
    )
endif(BLAS_FOUND)

set(CXX_STANDARD 23)

# define sk-plusplus target
add_library(${PROJECT_NAME} SHARED)
add_subdirectory(sk-plusplus/sk-plusplus/)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -g)
target_include_directories(${PROJECT_NAME} PUBLIC sk-plusplus/ ${BLAS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})

# Download dependency for benchmarking target if needed
if(NOT benchmark_FOUND)
    set(BENCHMARK_ENABLE_TESTING NO)

    include(FetchContent)

    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG origin/main
        )

    FetchContent_MakeAvailable(googlebenchmark)
endif()

# define benchmarking target
add_executable(bench)
add_subdirectory(benchs/)
target_compile_options(bench PRIVATE -Wall -Wextra -Werror -pedantic -g)
target_include_directories(bench PUBLIC benchs/)
target_link_libraries(bench PRIVATE ${PROJECT_NAME} benchmark::benchmark)

# Download dependency for testsuite target if needed
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# define testsuite target
add_executable(test)
add_subdirectory(tests/)
target_compile_options(test PRIVATE -Wall -Wextra -Werror -pedantic -g)
target_include_directories(test PUBLIC tests/)
target_link_libraries(test PRIVATE ${PROJECT_NAME} GTest::gtest GTest::gtest_main)
