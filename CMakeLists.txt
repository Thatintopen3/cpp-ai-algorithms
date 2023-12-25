
cmake_minimum_required(VERSION 3.10)
project(cpp_ai_algorithms)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add executable
add_executable(cpp_ai_algorithms ${SOURCES})

# Link libraries (e.g., Eigen for linear algebra)
# find_package(Eigen3 CONFIG REQUIRED)
# target_link_libraries(cpp_ai_algorithms Eigen3::Eigen)
