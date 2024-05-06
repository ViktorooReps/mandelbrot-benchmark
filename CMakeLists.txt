cmake_minimum_required(VERSION 3.11)
project(mandelbrot_benchmark C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "-march=native")

# Add the main executable
add_executable(mandelbrot_benchmark main.c)

# Add a custom target to generate the assembly output for main.c
set(flags_arg "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
separate_arguments(flags_arg)
add_custom_target(
        generate_assembly ALL
        COMMAND ${CMAKE_C_COMPILER} ${flags_arg} -S -o ${CMAKE_BINARY_DIR}/main.s ${CMAKE_SOURCE_DIR}/main.c
        DEPENDS main.c
)
# Ensure that the custom target is built after the main executable
add_dependencies(mandelbrot_benchmark generate_assembly)
