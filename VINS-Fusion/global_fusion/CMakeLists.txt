cmake_minimum_required(VERSION 2.8.3)
project(global_fusion_dyn)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

# find_package(Ceres REQUIRED)
include_directories("/home/chen/VINS/lib/include/")
set(Ceres_DIR /home/chen/VINS/solver/lib/cmake/Ceres/)
find_package(Ceres 1.14.0 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_subdirectory(./ThirdParty/GeographicLib/)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ./ThirdParty/GeographicLib/include/
)


catkin_package()


add_executable(global_fusion_dyn_node
	src/globalOptNode.cpp
	src/globalOpt.cpp)

target_link_libraries(global_fusion_dyn_node ${catkin_LIBRARIES} ${CERES_LIBRARIES} libGeographiccc)