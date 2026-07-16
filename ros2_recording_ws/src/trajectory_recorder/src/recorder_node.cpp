#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <cmath>

// ROS 2 核心与基础消息
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

// 图像处理
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

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
        // 在 recorder_node.cpp 的构造函数中：
        this->declare_parameter<std::string>("bota_csv_path", "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/bota_100hz.csv");
        this->declare_parameter<std::string>("gripper_csv_path", "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/gripper_10hz.csv");
        this->declare_parameter<std::string>("video_time_csv_path", "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/video_timestamps.csv");
        this->declare_parameter<std::string>("video_path", "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/camera.avi");

        bota_csv_path_ = this->get_parameter("bota_csv_path").as_string();
        gripper_csv_path_ = this->get_parameter("gripper_csv_path").as_string();
        video_time_csv_path_ = this->get_parameter("video_time_csv_path").as_string();
        video_path_ = this->get_parameter("video_path").as_string();

        frame_counter_ = 0;
        is_recording_ = false;

        bota_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "/bota_payload_compensator/wrench", 50, std::bind(&TrajectoryRecorder::botaCallback, this, std::placeholders::_1));

        gripper_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/robotiq/joint_states", 10, std::bind(&TrajectoryRecorder::gripperCallback, this, std::placeholders::_1));

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 10, std::bind(&TrajectoryRecorder::imageCallback, this, std::placeholders::_1));

        input_thread_ = std::thread(&TrajectoryRecorder::keyboardListener, this);

        std::cout << "ROS2 Recorder Node Ready. Press [ENTER] to start synchronous recording..." << std::endl;
    }

    ~TrajectoryRecorder() {
        if (input_thread_.joinable()) {
            input_thread_.join();
        }
        saveAllDataToFiles();
        if (video_writer_.isOpened()) {
            video_writer_.release();
            std::cout << "Video stream released" << std::endl;
        }
    }

private:
    void keyboardListener() {
        std::cin.get(); 
        std::cout << "Recording started" << std::endl;
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
                    frame.effort = std::isnan(msg->effort[i]) ? 0.0 : msg->effort[i];
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
            std::cerr << "cv_bridge exception: " << e.what() << std::endl;
        }
    }

    void saveAllDataToFiles() {
        if (!is_recording_ || (bota_buffer_.empty() && gripper_buffer_.empty())) {
            std::cerr << "Warning: No valid data captured" << std::endl;
            return;
        }

        if (!bota_buffer_.empty()) {
            std::lock_guard<std::mutex> lock(bota_mutex_);
            std::ofstream file(bota_csv_path_);
            file << "timestamp_us,F_x,F_y,F_z,tau_x,tau_y,tau_z\n";
            for (const auto& frame : bota_buffer_) {
                file << frame.timestamp_us << ","
                     << frame.fx << "," << frame.fy << "," << frame.fz << ","
                     << frame.tx << "," << frame.ty << "," << frame.tz << "\n"; 
            }
            file.close();
            std::cout << "Saved bota_100hz.csv (" << bota_buffer_.size() << " frames)" << std::endl;
        }

        if (!gripper_buffer_.empty()) {
            std::lock_guard<std::mutex> lock(gripper_mutex_);
            std::ofstream file(gripper_csv_path_);
            file << "timestamp_us,position,effort\n";
            for (const auto& frame : gripper_buffer_) {
                file << frame.timestamp_us << "," << frame.position << "," << frame.effort << "\n";
            }
            file.close();
            std::cout << "Saved gripper_10hz.csv (" << gripper_buffer_.size() << " frames)" << std::endl;
        }

        if (!video_time_buffer_.empty()) {
            std::lock_guard<std::mutex> lock(video_time_mutex_);
            std::ofstream file(video_time_csv_path_);
            file << "frame_index,timestamp_us\n";
            for (const auto& frame : video_time_buffer_) {
                file << frame.frame_index << "," << frame.timestamp_us << "\n";
            }
            file.close();
            std::cout << "Saved video_timestamps.csv (" << video_time_buffer_.size() << " frames)" << std::endl;
        }
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
