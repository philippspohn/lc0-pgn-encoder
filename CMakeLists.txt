cmake_minimum_required(VERSION 3.0)
project(lc0_pgn_encoder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_definitions(-DNO_PEXT)

include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cc ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.h)
add_executable(lc0_pgn_encoder ${SRC_FILES})
