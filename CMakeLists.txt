cmake_minimum_required(VERSION 3.5)
project(DirectMethod_cpp_demo)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# 在find_package前面
#set(cv_bridge_DIR /usr/local/share/cv_bridge_opencv3.4.17/share/cv_bridge/cmake) 
set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)  #在find_package前面

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

# set(CMAKE_BUILD_TYPE "Release") # copy from ch8
# add_definitions("-DENABLE_SSE")
# set(CMAKE_CXX_FLAGS "-std=c++14 ${SSE_FLAGS} -g -O3 -march=native")


### copy from orbslam2
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ") # copy from orbslam2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# cpp14 for ros2-foxy
# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC14)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

# LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
### end copy from orbslam2

# # Default to C++14
# if(NOT CMAKE_CXX_STANDARD)
#   set(CMAKE_CXX_STANDARD 14)
# endif()

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
# find_package(Pangolin REQUIRED)
find_package(ORB_SLAM2 REQUIRED)
# find_package(Sophus REQUIRED) # add by CW
# find_package(Eigen3 3.1.0 REQUIRED)
# FIND_PACKAGE(OpenCV 3 REQUIRED)
# find_package(G2O)

# find_package(OpenCV 3.0 QUIET)
# if(NOT OpenCV_FOUND)
#    find_package(OpenCV 2.4.3 QUIET)
#    if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#    endif()
# endif()
# add to check opencv version
# message(STATUS "OpenCV_DIR = ${OpenCV_DIR}")
# message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
# #message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

# include_directories(
# ${PROJECT_SOURCE_DIR}
# ${PROJECT_SOURCE_DIR}/include
# "/usr/include/eigen3/"
# ${Pangolin_INCLUDE_DIRS}
# ${EIGEN3_INCLUDE_DIR}
# ${OpenCV_INCLUDE_DIRS} # add by CW
# ${G2O_INCLUDE_DIRS}
# ${Sophus_INCLUDE_DIRS}
# )

# set( G2O_LIBS 
#     g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension 
# )

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)

# add_library(DirectMethod_VO SHARED
#         src/DirectMethod_cpp_demo/DireceMethodVO/System.cpp
#         src/DirectMethod_cpp_demo/DireceMethodVO/Tracking.cpp
#       #   Converter.cc
#         # visualization/Viewer.cc
#         )
# target_link_libraries(DirectMethod_VO 
#     ${OpenCV_LIBS} 
#     ${G2O_LIBS}
#     # ${Pangolin_INCLUDE_DIRS} # ?
#     )

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

add_executable(dm
  src/DirectMethod_cpp_demo/dm_main.cpp
  src/DirectMethod_cpp_demo/dm_vo_node.cpp
)

# target_link_libraries( ${PROJECT_NAME} 
#    #  ${realsense2_LIBRARY}
#     ${OpenCV_LIBS}
#    #  DirectMethod_VO
#     )

ament_target_dependencies(dm
   rclcpp 
   sensor_msgs 
   cv_bridge
   message_filters
   # ${OpenCV_LIBS}
   # ${G2O_LIBS}
   # DirectMethod_VO
   ORB_SLAM2 
#  Pangolin
)

# add_executable(mono 
#   src/monocular/mono.cpp
#   src/monocular/monocular-slam-node.cpp
# )
# ament_target_dependencies(mono rclcpp sensor_msgs cv_bridge ORB_SLAM2 Pangolin)

# add_executable(rgbd 
#   src/rgbd/rgbd.cpp
#   src/rgbd/rgbd-slam-node.cpp
# )
# ament_target_dependencies(rgbd rclcpp sensor_msgs cv_bridge message_filters ORB_SLAM2 Pangolin)

# add_executable(stereo 
#   src/stereo/stereo.cpp
#   src/stereo/stereo-slam-node.cpp
# )
# ament_target_dependencies(stereo rclcpp sensor_msgs cv_bridge message_filters ORB_SLAM2 Pangolin)

install(TARGETS
  dm
  # mono 
  # rgbd 
  # stereo
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
