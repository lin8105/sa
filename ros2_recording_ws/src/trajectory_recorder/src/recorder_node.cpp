#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <poll.h>
#include <string>
#include <stdexcept>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

// ROS 2 core and messages
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

// Image processing
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


struct BotaFrame {
    uint64_t timestamp_us;
    double fx;
    double fy;
    double fz;
    double tx;
    double ty;
    double tz;
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


struct SegmentFrame {
    std::size_t segment_index;
    uint64_t start_timestamp_us;
    uint64_t end_timestamp_us_exclusive;
    double start_relative_time_s;
    double end_relative_time_s;
    std::string label;
};


struct ActiveSegment {
    uint64_t start_timestamp_us;
    double start_relative_time_s;
    std::string label;
};


/**
 * Restores the terminal settings automatically when the keyboard thread exits.
 */
class TerminalSettingsGuard {
public:
    TerminalSettingsGuard()
        : valid_(false)
    {
    }

    TerminalSettingsGuard(const TerminalSettingsGuard&) = delete;
    TerminalSettingsGuard& operator=(const TerminalSettingsGuard&) = delete;

    bool enableImmediateInput()
    {
        if (!::isatty(STDIN_FILENO)) {
            std::cerr
                << "Warning: standard input is not a terminal. "
                << "Immediate single-key input is unavailable."
                << std::endl;
            return false;
        }

        if (::tcgetattr(STDIN_FILENO, &original_settings_) != 0) {
            std::cerr
                << "Failed to read terminal settings: "
                << std::strerror(errno)
                << std::endl;
            return false;
        }

        termios raw_settings = original_settings_;

        // Disable canonical mode and local echo.
        // Characters become available immediately without pressing Enter.
        raw_settings.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));

        // read() may return immediately. poll() controls the wait duration.
        raw_settings.c_cc[VMIN] = 0;
        raw_settings.c_cc[VTIME] = 0;

        if (::tcsetattr(STDIN_FILENO, TCSANOW, &raw_settings) != 0) {
            std::cerr
                << "Failed to enable immediate terminal input: "
                << std::strerror(errno)
                << std::endl;
            return false;
        }

        valid_ = true;
        return true;
    }

    void restore()
    {
        if (!valid_) {
            return;
        }

        if (::tcsetattr(STDIN_FILENO, TCSANOW, &original_settings_) != 0) {
            std::cerr
                << "Warning: failed to restore terminal settings: "
                << std::strerror(errno)
                << std::endl;
        }

        valid_ = false;
    }

    ~TerminalSettingsGuard()
    {
        restore();
    }

private:
    termios original_settings_{};
    bool valid_;
};


class TrajectoryRecorder : public rclcpp::Node {
public:
    TrajectoryRecorder()
        : Node("trajectory_recorder"),
          frame_counter_(0),
          is_recording_(false),
          input_thread_running_(true),
          stop_requested_(false),
          recording_start_timestamp_us_(0)
    {
        this->declare_parameter<std::string>(
            "bota_csv_path",
            "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/bota_100hz.csv"
        );
        this->declare_parameter<std::string>(
            "gripper_csv_path",
            "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/gripper_10hz.csv"
        );
        this->declare_parameter<std::string>(
            "video_time_csv_path",
            "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/video_timestamps.csv"
        );
        this->declare_parameter<std::string>(
            "video_path",
            "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/camera.avi"
        );
        this->declare_parameter<std::string>(
            "segment_csv_path",
            "/home/yue/Documents/zsc_Franka/ros2_recording_ws/data/segments.csv"
        );

        bota_csv_path_ =
            this->get_parameter("bota_csv_path").as_string();
        gripper_csv_path_ =
            this->get_parameter("gripper_csv_path").as_string();
        video_time_csv_path_ =
            this->get_parameter("video_time_csv_path").as_string();
        video_path_ =
            this->get_parameter("video_path").as_string();
        segment_csv_path_ =
            this->get_parameter("segment_csv_path").as_string();

        bota_sub_ =
            this->create_subscription<geometry_msgs::msg::WrenchStamped>(
                "/bota_payload_compensator/wrench",
                50,
                std::bind(
                    &TrajectoryRecorder::botaCallback,
                    this,
                    std::placeholders::_1
                )
            );

        gripper_sub_ =
            this->create_subscription<sensor_msgs::msg::JointState>(
                "/robotiq/joint_states",
                10,
                std::bind(
                    &TrajectoryRecorder::gripperCallback,
                    this,
                    std::placeholders::_1
                )
            );

        image_sub_ =
            this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/color/image_raw",
                10,
                std::bind(
                    &TrajectoryRecorder::imageCallback,
                    this,
                    std::placeholders::_1
                )
            );

        input_thread_ =
            std::thread(&TrajectoryRecorder::keyboardListener, this);

        std::cout
            << "ROS 2 recorder node ready." << std::endl
            << "Press [ENTER] to start recording." << std::endl
            << "Press [s] to generate an annotation." << std::endl
            << "Press [ENTER] to close the final segment, save, and exit."
            << std::endl;
    }

    ~TrajectoryRecorder() override
    {
        // The keyboard thread uses poll() with a timeout, so setting this flag
        // is sufficient to let it leave without waiting for another key press.
        input_thread_running_.store(false);

        if (input_thread_.joinable()) {
            input_thread_.join();
        }

        finalizeActiveSegment(currentTimestampUs());
        saveAllDataToFiles();

        if (video_writer_.isOpened()) {
            video_writer_.release();
            std::cout << "Video stream released." << std::endl;
        }
    }

private:
    uint64_t currentTimestampUs() const
    {
        return static_cast<uint64_t>(
            this->now().nanoseconds() / 1000
        );
    }

    void keyboardListener()
    {
        TerminalSettingsGuard terminal_guard;

        try {
            if (!terminal_guard.enableImmediateInput()) {
                std::cerr
                    << "Keyboard listener stopped because immediate input "
                    << "could not be enabled."
                    << std::endl;
                return;
            }

            while (input_thread_running_.load()) {
                pollfd input_poll{};
                input_poll.fd = STDIN_FILENO;
                input_poll.events = POLLIN;

                // A short timeout ensures that the destructor can stop and
                // join this thread even when the user presses no key.
                const int poll_result = ::poll(&input_poll, 1, 100);

                if (poll_result < 0) {
                    if (errno == EINTR) {
                        continue;
                    }

                    std::cerr
                        << "Keyboard poll failed: "
                        << std::strerror(errno)
                        << std::endl;
                    break;
                }

                if (poll_result == 0) {
                    continue;
                }

                if ((input_poll.revents & POLLIN) == 0) {
                    continue;
                }

                char key = '\0';
                const ssize_t bytes_read =
                    ::read(STDIN_FILENO, &key, sizeof(key));

                if (bytes_read <= 0) {
                    continue;
                }

                if (!is_recording_.load()) {
                    if (key == '\n' || key == '\r') {
                        startRecording();
                    }

                    continue;
                }

                if (key == 's' || key == 'S') {
                    recordBoundaryAndStartNewSegment();
                } else if (key == '\n' || key == '\r') {
                    requestStop();
                    break;
                }
            }
        } catch (const std::exception& exception) {
            std::cerr
                << "Keyboard listener exception: "
                << exception.what()
                << std::endl;
        } catch (...) {
            std::cerr
                << "Keyboard listener stopped after an unknown exception."
                << std::endl;
        }

        // The guard restores the terminal here, including exception paths.
    }

    void startRecording()
    {
        if (is_recording_.load()) {
            return;
        }

        const uint64_t start_timestamp_us = currentTimestampUs();
        recording_start_timestamp_us_.store(start_timestamp_us);

        {
            std::lock_guard<std::mutex> lock(segment_mutex_);
            active_segment_.start_timestamp_us = start_timestamp_us;
            active_segment_.start_relative_time_s = 0.0;
            active_segment_.label.clear();
            has_active_segment_ = true;
        }

        // Enable callbacks only after the recording start timestamp and the
        // first empty-label segment have both been initialized.
        is_recording_.store(true);

        std::cout
            << std::endl
            << "Recording started at "
            << start_timestamp_us
            << " us." << std::endl
            << "Press [s] to generate an annotation." << std::endl
            << "Press [ENTER] to close the final segment, save, and exit."
            << std::endl;
    }

    void finalizeActiveSegment(uint64_t end_timestamp_us_exclusive)
    {
        std::lock_guard<std::mutex> lock(segment_mutex_);

        if (!has_active_segment_) {
            return;
        }

        const uint64_t recording_start_timestamp_us =
            recording_start_timestamp_us_.load();

        if (
            recording_start_timestamp_us == 0
            || end_timestamp_us_exclusive
                <= active_segment_.start_timestamp_us
        ) {
            return;
        }

        SegmentFrame segment{};
        segment.segment_index = segment_buffer_.size();
        segment.start_timestamp_us =
            active_segment_.start_timestamp_us;
        segment.end_timestamp_us_exclusive =
            end_timestamp_us_exclusive;
        segment.start_relative_time_s =
            active_segment_.start_relative_time_s;
        segment.end_relative_time_s =
            static_cast<double>(
                end_timestamp_us_exclusive
                - recording_start_timestamp_us
            ) /
            1'000'000.0;
        segment.label = active_segment_.label;

        segment_buffer_.push_back(segment);
        has_active_segment_ = false;
    }

    void recordBoundaryAndStartNewSegment()
    {
        if (!is_recording_.load()) {
            return;
        }

        // Capture the boundary immediately when [s] is detected. No label
        // input or other blocking operation occurs before this timestamp.
        const uint64_t boundary_timestamp_us = currentTimestampUs();
        const uint64_t recording_start_timestamp_us =
            recording_start_timestamp_us_.load();

        if (recording_start_timestamp_us == 0) {
            std::cerr
                << "Warning: recording start timestamp is unavailable. "
                << "Boundary was not recorded."
                << std::endl;
            return;
        }

        const double boundary_relative_time_s =
            static_cast<double>(
                boundary_timestamp_us - recording_start_timestamp_us
            ) /
            1'000'000.0;

        std::size_t completed_segment_index = 0;
        std::size_t new_segment_index = 0;

        {
            std::lock_guard<std::mutex> lock(segment_mutex_);

            if (!has_active_segment_) {
                std::cerr
                    << "Warning: no active segment exists."
                    << std::endl;
                return;
            }

            // Ignore a duplicate boundary that would create a zero-duration
            // segment, for example from key repeat or an immediate second [s].
            if (boundary_timestamp_us <= active_segment_.start_timestamp_us) {
                std::cerr
                    << "Warning: ignored a non-increasing segment boundary."
                    << std::endl;
                return;
            }

            SegmentFrame segment{};
            segment.segment_index = segment_buffer_.size();
            segment.start_timestamp_us =
                active_segment_.start_timestamp_us;
            segment.end_timestamp_us_exclusive =
                boundary_timestamp_us;
            segment.start_relative_time_s =
                active_segment_.start_relative_time_s;
            segment.end_relative_time_s =
                boundary_relative_time_s;
            segment.label.clear();

            completed_segment_index = segment.segment_index;
            segment_buffer_.push_back(segment);

            active_segment_.start_timestamp_us =
                boundary_timestamp_us;
            active_segment_.start_relative_time_s =
                boundary_relative_time_s;
            active_segment_.label.clear();
            has_active_segment_ = true;
            new_segment_index = segment_buffer_.size();
        }

        std::cout
            << std::endl
            << "Marked boundary at relative time "
            << std::fixed
            << std::setprecision(6)
            << boundary_relative_time_s
            << " s. Closed segment "
            << completed_segment_index
            << " and started segment "
            << new_segment_index
            << " with an empty label."
            << std::defaultfloat
            << std::endl;
    }

    void requestStop()
    {
        bool expected = false;

        if (!stop_requested_.compare_exchange_strong(expected, true)) {
            return;
        }

        finalizeActiveSegment(currentTimestampUs());
        is_recording_.store(false);
        input_thread_running_.store(false);

        std::cout
            << std::endl
            << "Stop requested. Saving data..."
            << std::endl;

        // rclcpp::shutdown() is thread-safe and causes rclcpp::spin() in main
        // to return. The node destructor then joins this keyboard thread and
        // saves all buffered data.
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
    }

    void botaCallback(
        const geometry_msgs::msg::WrenchStamped::SharedPtr msg
    )
    {
        if (!is_recording_.load()) {
            return;
        }

        BotaFrame frame{};
        frame.timestamp_us = currentTimestampUs();
        frame.fx = msg->wrench.force.x;
        frame.fy = msg->wrench.force.y;
        frame.fz = msg->wrench.force.z;
        frame.tx = msg->wrench.torque.x;
        frame.ty = msg->wrench.torque.y;
        frame.tz = msg->wrench.torque.z;

        std::lock_guard<std::mutex> lock(bota_mutex_);
        bota_buffer_.push_back(frame);
    }

    void gripperCallback(
        const sensor_msgs::msg::JointState::SharedPtr msg
    )
    {
        if (!is_recording_.load()) {
            return;
        }

        GripperFrame frame{};
        frame.timestamp_us = currentTimestampUs();
        frame.position = 0.0;
        frame.effort = 0.0;

        for (std::size_t i = 0; i < msg->name.size(); ++i) {
            if (
                msg->name[i]
                == "robotiq_hande_left_finger_joint"
            ) {
                if (msg->position.size() > i) {
                    frame.position = msg->position[i];
                }

                if (msg->effort.size() > i) {
                    frame.effort =
                        std::isnan(msg->effort[i])
                            ? 0.0
                            : msg->effort[i];
                }

                break;
            }
        }

        std::lock_guard<std::mutex> lock(gripper_mutex_);
        gripper_buffer_.push_back(frame);
    }

    void imageCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& msg
    )
    {
        if (!is_recording_.load()) {
            return;
        }

        try {
            cv_bridge::CvImagePtr cv_ptr =
                cv_bridge::toCvCopy(
                    msg,
                    sensor_msgs::image_encodings::BGR8
                );

            const cv::Mat& frame = cv_ptr->image;

            if (!video_writer_.isOpened()) {
                video_writer_.open(
                    video_path_,
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    30.0,
                    frame.size(),
                    true
                );

                if (!video_writer_.isOpened()) {
                    std::cerr
                        << "Failed to open video output: "
                        << video_path_
                        << std::endl;
                    return;
                }
            }

            video_writer_.write(frame);

            VideoTimeFrame time_frame{};
            time_frame.frame_index = frame_counter_++;
            time_frame.timestamp_us = currentTimestampUs();

            std::lock_guard<std::mutex> lock(video_time_mutex_);
            video_time_buffer_.push_back(time_frame);

        } catch (const cv_bridge::Exception& exception) {
            std::cerr
                << "cv_bridge exception: "
                << exception.what()
                << std::endl;
        }
    }

    void saveAllDataToFiles()
    {
        bool saved_anything = false;

        {
            std::lock_guard<std::mutex> lock(bota_mutex_);

            if (!bota_buffer_.empty()) {
                std::ofstream file(bota_csv_path_);

                if (!file.is_open()) {
                    std::cerr
                        << "Failed to open Bota output: "
                        << bota_csv_path_
                        << std::endl;
                } else {
                    file
                        << "timestamp_us,F_x,F_y,F_z,"
                        << "tau_x,tau_y,tau_z\n";

                    for (const auto& frame : bota_buffer_) {
                        file
                            << frame.timestamp_us << ","
                            << frame.fx << ","
                            << frame.fy << ","
                            << frame.fz << ","
                            << frame.tx << ","
                            << frame.ty << ","
                            << frame.tz << "\n";
                    }

                    std::cout
                        << "Saved "
                        << bota_csv_path_
                        << " ("
                        << bota_buffer_.size()
                        << " samples)."
                        << std::endl;

                    saved_anything = true;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(gripper_mutex_);

            if (!gripper_buffer_.empty()) {
                std::ofstream file(gripper_csv_path_);

                if (!file.is_open()) {
                    std::cerr
                        << "Failed to open gripper output: "
                        << gripper_csv_path_
                        << std::endl;
                } else {
                    file << "timestamp_us,position,effort\n";

                    for (const auto& frame : gripper_buffer_) {
                        file
                            << frame.timestamp_us << ","
                            << frame.position << ","
                            << frame.effort << "\n";
                    }

                    std::cout
                        << "Saved "
                        << gripper_csv_path_
                        << " ("
                        << gripper_buffer_.size()
                        << " samples)."
                        << std::endl;

                    saved_anything = true;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(video_time_mutex_);

            if (!video_time_buffer_.empty()) {
                std::ofstream file(video_time_csv_path_);

                if (!file.is_open()) {
                    std::cerr
                        << "Failed to open video timestamp output: "
                        << video_time_csv_path_
                        << std::endl;
                } else {
                    file << "frame_index,timestamp_us\n";

                    for (const auto& frame : video_time_buffer_) {
                        file
                            << frame.frame_index << ","
                            << frame.timestamp_us << "\n";
                    }

                    std::cout
                        << "Saved "
                        << video_time_csv_path_
                        << " ("
                        << video_time_buffer_.size()
                        << " frames)."
                        << std::endl;

                    saved_anything = true;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(segment_mutex_);

            if (!segment_buffer_.empty()) {
                std::ofstream file(segment_csv_path_);

                if (!file.is_open()) {
                    std::cerr
                        << "Failed to open segment output: "
                        << segment_csv_path_
                        << std::endl;
                } else {
                    file
                        << "segment_index,start_timestamp_us,"
                        << "end_timestamp_us_exclusive,"
                        << "start_relative_time_s,"
                        << "end_relative_time_s,label\n";

                    file << std::fixed << std::setprecision(6);

                    for (const auto& segment : segment_buffer_) {
                        file
                            << segment.segment_index << ","
                            << segment.start_timestamp_us << ","
                            << segment.end_timestamp_us_exclusive << ","
                            << segment.start_relative_time_s << ","
                            << segment.end_relative_time_s << ","
                            << segment.label << "\n";
                    }

                    std::cout
                        << "Saved "
                        << segment_csv_path_
                        << " ("
                        << segment_buffer_.size()
                        << " closed segments)."
                        << std::endl;

                    saved_anything = true;
                }
            }
        }

        if (!saved_anything) {
            std::cerr
                << "Warning: no Bota, gripper, video timestamp, "
                << "or segment data was captured."
                << std::endl;
        }
    }

    rclcpp::Subscription<
        geometry_msgs::msg::WrenchStamped
    >::SharedPtr bota_sub_;

    rclcpp::Subscription<
        sensor_msgs::msg::JointState
    >::SharedPtr gripper_sub_;

    rclcpp::Subscription<
        sensor_msgs::msg::Image
    >::SharedPtr image_sub_;

    std::thread input_thread_;
    std::atomic<bool> is_recording_;
    std::atomic<bool> input_thread_running_;
    std::atomic<bool> stop_requested_;
    std::atomic<uint64_t> recording_start_timestamp_us_;

    std::vector<BotaFrame> bota_buffer_;
    std::mutex bota_mutex_;

    std::vector<GripperFrame> gripper_buffer_;
    std::mutex gripper_mutex_;

    std::vector<VideoTimeFrame> video_time_buffer_;
    std::mutex video_time_mutex_;

    std::vector<SegmentFrame> segment_buffer_;
    ActiveSegment active_segment_{};
    bool has_active_segment_{false};
    std::mutex segment_mutex_;

    std::string bota_csv_path_;
    std::string gripper_csv_path_;
    std::string video_time_csv_path_;
    std::string video_path_;
    std::string segment_csv_path_;

    int frame_counter_;
    cv::VideoWriter video_writer_;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<TrajectoryRecorder>();
        rclcpp::spin(node);

        // The keyboard thread normally calls shutdown after q. This covers
        // external shutdown paths such as Ctrl+C as well.
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
    } catch (const std::exception& exception) {
        std::cerr
            << "Recorder exception: "
            << exception.what()
            << std::endl;

        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }

        return 1;
    }

    return 0;
}
