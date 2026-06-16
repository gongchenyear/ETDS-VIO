cmake_minimum_required(VERSION 2.8.3)
project(vins_drt_dyn)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models_dyn
    image_transport)

# find_package(OpenCV 4 REQUIRED)
find_package(OpenCV 4 REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

# find_package(Ceres REQUIRED)
include_directories("/home/chen/VINS/lib/include/")
set(Ceres_DIR /home/chen/VINS/solver/lib/cmake/Ceres/)
find_package(Ceres 1.14.0 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

# DRT 不再需要 Pangolin（已注释掉可视化代码）

# 尝试在系统中查找 Sophus，如果找不到则使用本地路径
find_package(Sophus QUIET)
if(NOT Sophus_FOUND)
  set(Sophus_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../drt-vio-init/thirdparty/Sophus)
  if(NOT EXISTS ${Sophus_INCLUDE_DIR})
    # 备选本地路径
    set(Sophus_INCLUDE_DIR /home/chen/drt-vio-init/thirdparty/Sophus)
  endif()
endif()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src/drt_init/include         # 本地 DRT 头文件
  ${Sophus_INCLUDE_DIR}                               # Sophus 库（系统或本地）
)

catkin_package()

add_library(vins_drt_dyn_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/estimator/drt_adapter.cpp
    src/drt_init/drtVioInit.cpp                        # DRT 核心代码
    src/drt_init/drtTightlyCoupled.cpp                 # DRT 紧耦合初始化
    src/drt_init/polynomial.cc                         # DRT 多项式求解
    src/drt_init/imuPreintegrated.cpp                  # DRT IMU 预积分
    src/drt_init/basicTypes.cpp                        # DRT 基础类型
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp)
target_link_libraries(vins_drt_dyn_lib 
  ${catkin_LIBRARIES} 
  ${OpenCV_LIBS} 
  ${CERES_LIBRARIES}
  # 不再链接外部 DRT 库，使用本地编译的代码
)


add_executable(vins_drt_dyn_node src/rosNodeTest.cpp)
target_link_libraries(vins_drt_dyn_node vins_drt_dyn_lib)

add_executable(kitti_odom_test_drt_dyn src/KITTIOdomTest.cpp)
target_link_libraries(kitti_odom_test_drt_dyn vins_drt_dyn_lib)

add_executable(kitti_gps_test_drt_dyn src/KITTIGPSTest.cpp)
target_link_libraries(kitti_gps_test_drt_dyn vins_drt_dyn_lib)
