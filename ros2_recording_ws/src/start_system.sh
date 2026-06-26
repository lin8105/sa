#!/bin/bash

# 定义你的各个物理工作空间路径
WS_PATH="/home/yue/Documents/zsc_Franka/ros2_recording_ws"
ORBBEC_PATH="/home/yue/Documents/zsc_Franka/OrbbecSDK_ROS2"
ROBOTIQ_PATH="/home/yue/Documents/zsc_Franka/robotiq/ros2_ws"
BOTA_PATH="/home/yue/Documents/zsc_Franka/bota"

# 一劳永逸：先在主脚本的环境里注入变量（部分子终端会继承）
export ROS_DOMAIN_ID=1

# 1. 启动容器（如果已经退出了）并启动内部的 Franka 底层驱动真机节点 
# 【注意】：Docker 内部执行也必须显式加上 export ROS_DOMAIN_ID=1
docker start multipanda-container
gnome-terminal --tab --title="Franka Docker Driver" -- bash -c "docker exec -it multipanda-container bash -c 'export ROS_DOMAIN_ID=1 && source ~/multipanda_ws/install/setup.bash && ros2 launch franka_bringup franka.launch.py robot_ip:=192.168.3.100'; exec bash"

# 2. 启动 相机节点 (加上 Domain ID)
gnome-terminal --tab --title="Camera Node" -- bash -c "cd $ORBBEC_PATH && export ROS_DOMAIN_ID=1 && source install/setup.bash && ros2 launch orbbec_camera femto_mega.launch.py; exec bash"

# 3. 启动 Robotiq 夹爪 (加上 Domain ID)
gnome-terminal --tab --title="Robotiq Gripper" -- bash -c "cd $ROBOTIQ_PATH && export ROS_DOMAIN_ID=1 && source install/setup.bash && ros2 launch robotiq_hande_driver gripper_controller_preview.launch.py use_fake_hardware:=false tty_port:=/dev/ttyUSB0; exec bash"

# 4. 启动 bota (两个标签页独立运行，并且均加上 Domain ID)
gnome-terminal --tab --title="1-Bota Driver" -- bash -c " \
  cd \$BOTA_PATH && \
  export ROS_DOMAIN_ID=1 && \
  source install/setup.bash && \
  ros2 run bota_driver bota_driver_node --ros-args -p node_name:=bota_ft_sensor -p config_file:=\"/home/yue/Documents/zsc_Franka/bota/src/bota_driver_ros2_example/bota_config/ethercat_gen0.json\" -p output_rate:=100.0; \
  exec bash"

gnome-terminal --tab --title="2-Bota Compensator" -- bash -c " \
  cd \$BOTA_PATH && \
  export ROS_DOMAIN_ID=1 && \
  source install/setup.bash && \
  ros2 run bota_payload_utils bota_payload_compensator_node --ros-args -p node_name:=bota_payload_compensator -p input_node_name:=bota_ft_sensor -p config_file:=\"/home/yue/Documents/zsc_Franka/bota/src/bota_driver_ros2_example/bota_config/payload_config.json\"; \
  exec bash"

# 给高频硬件传感器广播留出数据落盘就绪时间
sleep 1

# 5. 启动你的中央同步轨迹记录节点 (加上 Domain ID)
gnome-terminal --tab --title="Trajectory Recorder" -- bash -c "cd $WS_PATH && export ROS_DOMAIN_ID=1 && source install/setup.bash && ros2 run trajectory_recorder recorder_node; exec bash"
