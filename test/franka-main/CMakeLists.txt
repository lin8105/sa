cmake_minimum_required(VERSION 3.10)
project(Multimodal_Trajectory_Recorder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Franka REQUIRED)
find_package(Threads REQUIRED)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_executable(path_recorder
    path_recorder_controller.cpp
    common/examples_common.cpp
    FrankaRobot.cpp
)

target_link_libraries(path_recorder
    PRIVATE
    Franka::Franka
    Eigen3::Eigen
    Threads::Threads
    dqrobotics 
)