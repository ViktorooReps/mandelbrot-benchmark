cmake_minimum_required(VERSION 3.11)
project(mandelbrot_benchmark C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS -march=native)
set(CMAKE_C_FLAGS_RELEASE -O3 -DNDEBUG -march=native)

#set(CMAKE_C_FLAGS_RELEASE -O3 -DNDEBUG)

# Add the main executable
add_executable(mandelbrot_benchmark main.c)

# Add a custom target to generate the assembly output for main.c
add_custom_target(
        generate_assembly ALL
        COMMAND ${CMAKE_C_COMPILER} ${CMAKE_C_FLAGS_RELEASE} -S -o ${CMAKE_BINARY_DIR}/main.s ${CMAKE_SOURCE_DIR}/main.c
        DEPENDS main.c
)

# Ensure that the custom target is built after the main executable
add_dependencies(mandelbrot_benchmark generate_assembly)
