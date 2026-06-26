#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>

// ROS 2 核心与基础消息
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

// 图像处理
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// 独立数据缓存结构体
struct BotaFrame {
    uint64_t timestamp_us;
    double fx, fy, fz;
    double tx, ty, tz;
};

struct GripperFrame {
    uint64_t timestamp_us;
    double position;
    double effort;
};

struct VideoTimeFrame {
    int frame_index;
    uint64_t timestamp_us;
};

class TrajectoryRecorder : public rclcpp::Node {
public:
    TrajectoryRecorder() : Node("trajectory_recorder") {
        // 声明并获取分流保存的文件路径参数
        this->declare_parameter<std::string>("bota_csv_path", "bota_100hz.csv");
        this->declare_parameter<std::string>("gripper_csv_path", "gripper_10hz.csv");
        this->declare_parameter<std::string>("video_time_csv_path", "video_timestamps.csv");
        this->declare_parameter<std::string>("video_path", "camera.mp4");

        bota_csv_path_ = this->get_parameter("bota_csv_path").as_string();
        gripper_csv_path_ = this->get_parameter("gripper_csv_path").as_string();
        video_time_csv_path_ = this->get_parameter("video_time_csv_path").as_string();
        video_path_ = this->get_parameter("video_path").as_string();

        frame_counter_ = 0;
        is_recording_ = false; // 默认挂起，不录制数据

        // 1. 订阅 Bota 力传感器 100Hz 原始数据流
        bota_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "/bota_payload_compensator/wrench", 10, std::bind(&TrajectoryRecorder::botaCallback, this, std::placeholders::_1));

        // 2. 订阅 Robotiq 夹爪 10Hz 状态流
        gripper_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/robotiq/joint_states", 10, std::bind(&TrajectoryRecorder::gripperCallback, this, std::placeholders::_1));

        // 3. 订阅相机图像流
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10, std::bind(&TrajectoryRecorder::imageCallback, this, std::placeholders::_1));

        // 启动独立线程监听终端 Enter 按键输入
        input_thread_ = std::thread(&TrajectoryRecorder::keyboardListener, this);

        RCLCPP_INFO(this->get_logger(), "==========================================================");
        RCLCPP_INFO(this->get_logger(), "🤖 轨迹采集记录系统已就绪（Bota + Robotiq + Camera 同步模式）。");
        RCLCPP_INFO(this->get_logger(), "👉 准备好后，在当前终端按 [ENTER] 键开始同步录制传感器数据与视频...");
        RCLCPP_INFO(this->get_logger(), "==========================================================");
    }

    ~TrajectoryRecorder() {
        if (input_thread_.joinable()) {
            input_thread_.join();
        }
        saveAllDataToFiles();
        if (video_writer_.isOpened()) {
            video_writer_.release();
            RCLCPP_INFO(this->get_logger(), "🎥 视频文件编码流已安全关闭。");
        }
    }

private:
    void keyboardListener() {
        std::cin.get(); 
        RCLCPP_INFO(this->get_logger(), "🔴 [RECORDING] 检测到 ENTER 触发！录制通道开启，开始高速缓存数据...");
        is_recording_ = true; 
    }

    void botaCallback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
        if (!is_recording_) return;

        BotaFrame frame;
        frame.timestamp_us = this->now().nanoseconds() / 1000;
        frame.fx = msg->wrench.force.x;
        frame.fy = msg->wrench.force.y;
        frame.fz = msg->wrench.force.z;
        frame.tx = msg->wrench.torque.x;
        frame.ty = msg->wrench.torque.y;
        frame.tz = msg->wrench.torque.z;

        std::lock_guard<std::mutex> lock(bota_mutex_);
        bota_buffer_.push_back(frame);
    }

    void gripperCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (!is_recording_) return;

        GripperFrame frame;
        frame.timestamp_us = this->now().nanoseconds() / 1000;
        frame.position = 0.0;
        frame.effort = 0.0;

        for (size_t i = 0; i < msg->name.size(); ++i) {
            if (msg->name[i] == "robotiq_hande_left_finger_joint") {
                frame.position = msg->position[i];
                if (msg->effort.size() > i) {
                    frame.effort = msg->effort[i];
                }
                break;
            }
        }

        std::lock_guard<std::mutex> lock(gripper_mutex_);
        gripper_buffer_.push_back(frame);
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        if (!is_recording_) return;

        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat frame = cv_ptr->image;

            if (!video_writer_.isOpened()) {
                // 默认使用 MJPG 编码
                video_writer_.open(video_path_, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, frame.size(), true);
            }
            
            if (video_writer_.isOpened()) {
                video_writer_.write(frame);
                
                VideoTimeFrame time_frame;
                time_frame.frame_index = frame_counter_++;
                time_frame.timestamp_us = this->now().nanoseconds() / 1000;

                std::lock_guard<std::mutex> lock(video_time_mutex_);
                video_time_buffer_.push_back(time_frame);
            }
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge 录制图像异常: %s", e.what());
        }
    }

    void saveAllDataToFiles() {
        if (!is_recording_ || (bota_buffer_.empty() && gripper_buffer_.empty())) {
            RCLCPP_WARN(this->get_logger(), "⚠️ 节点未曾激活录制或未收到有效数据，不生成 any 文件。");
            return;
        }

        // --- 1. 存储 Bota 力矩传感器数据 (已剔除伪速度列) ---
        if (!bota_buffer_.empty()) {
            std::lock_guard<std::mutex> lock(bota_mutex_);
            RCLCPP_INFO(this->get_logger(), "💾 正在落盘 Bota 动力学数据 (%zu 帧)...", bota_buffer_.size());
            std::ofstream file(bota_csv_path_);
            
            // 纯净表头：剔除了多余的 v 和 w
            file << "timestamp_us,F_x,F_y,F_z,tau_x,tau_y,tau_z\n";

            for (const auto& frame : bota_buffer_) {
                file << frame.timestamp_us << ","
                     << frame.fx << "," << frame.fy << "," << frame.fz << ","
                     << frame.tx << "," << frame.ty << "," << frame.tz << "\n"; 
            }
            file.close();
        }

        // --- 2. 存储 Robotiq 夹爪数据 ---
        if (!gripper_buffer_.empty()) {
            std::lock_guard<std::mutex> lock(gripper_mutex_);
            RCLCPP_INFO(this->get_logger(), "💾 正在落盘 Robotiq 状态数据 (%zu 帧)...", gripper_buffer_.size());
            std::ofstream file(gripper_csv_path_);
            
            file << "timestamp_us,position,effort\n";
            for (const auto& frame : gripper_buffer_) {
                file << frame.timestamp_us << ","
                     << frame.position << ","
                     << frame.effort << "\n";
            }
            file.close();
        }

        // --- 3. 存储视频帧时间戳 ---
        if (!video_time_buffer_.empty()) {
            std::lock_guard<std::mutex> lock(video_time_mutex_);
            RCLCPP_INFO(this->get_logger(), "💾 正在落盘视频帧同步时间戳对照表 (%zu 帧)...", video_time_buffer_.size());
            std::ofstream file(video_time_csv_path_);
            
            file << "frame_index,timestamp_us\n";
            for (const auto& frame : video_time_buffer_) {
                file << frame.frame_index << "," << frame.timestamp_us << "\n";
            }
            file.close();
        }

        RCLCPP_INFO(this->get_logger(), "🎉 [成功] 各传感器离线示教文件及视频时间戳已安全落盘！");
    }

    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr bota_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr gripper_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    
    std::thread input_thread_;
    std::atomic<bool> is_recording_;

    std::vector<BotaFrame> bota_buffer_;
    std::mutex bota_mutex_;

    std::vector<GripperFrame> gripper_buffer_;
    std::mutex gripper_mutex_;

    std::vector<VideoTimeFrame> video_time_buffer_;
    std::mutex video_time_mutex_;

    std::string bota_csv_path_, gripper_csv_path_, video_time_csv_path_, video_path_;
    int frame_counter_;
    cv::VideoWriter video_writer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajectoryRecorder>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}