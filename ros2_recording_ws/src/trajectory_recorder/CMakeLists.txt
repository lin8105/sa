cmake_minimum_required(VERSION 3.8)
project(trajectory_recorder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    include
    src
    ${EIGEN3_INCLUDE_DIRS}
)

# 直接添加你的 node，不依赖任何第三方包查找
add_executable(recorder_node 
    src/recorder_node.cpp 
    src/cnpy.cpp
)

target_link_libraries(recorder_node PRIVATE 
    ${OpenCV_LIBS}
    rclcpp::rclcpp
    cv_bridge::cv_bridge
    message_filters::message_filters
    franka_msgs::franka_msgs__rosidl_typesupport_cpp
    pthread
    z
)

install(TARGETS recorder_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()