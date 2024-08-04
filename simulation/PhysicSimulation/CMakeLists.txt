cmake_minimum_required(VERSION 3.22)
project(PhysicSimulation)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fvisibility=hidden")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /fvisibility=hidden")
endif()


find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(PhysicSimulation main.cpp Config.h math/Vector2.cpp math/Vector2.h Simulation.cpp Simulation.h Wall.cpp Wall.h Hole.cpp Hole.h)
add_library(PhysicSimulationLib SHARED main.cpp Config.h math/Vector2.cpp math/Vector2.h Simulation.cpp Simulation.h Wall.cpp Wall.h Hole.cpp Hole.h)
add_library(PhysicSimulationNoGraphicsLib SHARED main.cpp Config.h math/Vector2.cpp math/Vector2.h Simulation.cpp Simulation.h Wall.cpp Wall.h Hole.cpp Hole.h)


# Enable optimizations
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(PhysicSimulation PRIVATE
            -O3
            -ffast-math
            -march=native
            -funroll-loops
            -finline-functions
            -fopt-info-vec-all
    )

    target_compile_options(PhysicSimulationLib PRIVATE
            -O3
            -ffast-math
            -march=native
            -funroll-loops
            -finline-functions
            -fopt-info-vec-all
    )

    target_compile_options(PhysicSimulationNoGraphicsLib PRIVATE
            -O3
            -ffast-math
            -funroll-loops
            -finline-functions
            -fopt-info-vec-all
    )
endif()


if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_link_options(PhysicSimulation PRIVATE -flto)
endif()

target_link_libraries(PhysicSimulation sfml-graphics sfml-window sfml-system OpenMP::OpenMP_CXX)
target_link_libraries(PhysicSimulationLib sfml-graphics sfml-window sfml-system OpenMP::OpenMP_CXX)
target_link_libraries(PhysicSimulationNoGraphicsLib OpenMP::OpenMP_CXX)
