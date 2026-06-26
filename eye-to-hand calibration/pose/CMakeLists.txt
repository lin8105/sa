cmake_minimum_required(VERSION 3.16)
project(franka_pose_recorder LANGUAGES CXX)

# 开启 C++17 标准（libfranka 底层强依赖）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================= 1. 寻找系统里安装的普通 libfranka 驱动 =================
find_package(Franka REQUIRED)

# ================= 2. 编译你的源文件 =================
# 如果它是想编译成一个独立运行的可执行程序（带 main 函数）：
add_executable(path_recorder_controller path_recorder_controller.cpp)

# 如果你其实想把它编译成一个普通动态链接库（.so 文件），就把上面那行注释掉，用下面这行：
# add_library(path_recorder_controller SHARED src/path_recorder_controller.cpp)

# ================= 3. 包含头文件路径 =================
# 包含你自己的 include 目录以及 libfranka 的头文件
target_include_directories(path_recorder_controller PRIVATE 
  include
  ${Franka_INCLUDE_DIRS}
)

# ================= 4. 链接 libfranka 底层库 =================
target_link_libraries(path_recorder_controller PRIVATE 
  Franka::Franka
)

# 显式链接系统线程库（libfranka 内部通信需要线程安全）
find_package(Threads REQUIRED)
target_link_libraries(path_recorder_controller PRIVATE Threads::Threads)
