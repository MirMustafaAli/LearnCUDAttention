cmake_minimum_required(VERSION 3.29)
project(LearnCUDAttention)

set(CMAKE_C_STANDARD 99)  # Assuming the C standard being used is C99

# Enable position-independent code generation (necessary for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add the library target (shared library)
add_library(attention SHARED attention.c)

# Link the math library
target_link_libraries(attention m)