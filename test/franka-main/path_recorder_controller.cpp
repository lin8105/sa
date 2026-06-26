
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/robot.h>
#include "common/examples_common.h"

// --- Logging Globals (Removed is_grasped) ---
std::atomic<double> shared_gripper_width{0.08};
std::atomic<bool> keep_running{true};

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname> <outfile.csv>" << std::endl;
        return -1;
    }

    // --- Declare outside 'try' for safe cleanup ---
    std::thread gripper_thread;
    std::ofstream csv_file;

    // Joint limits
    Eigen::Matrix<double, 7, 1> q_min;
    q_min << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973;
    Eigen::Matrix<double, 7, 1> q_max;
    q_max << 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973;
    Eigen::Matrix<double, 7, 1> q_mid = (q_min + q_max) / 2;

    // Compliance parameters
    const double translational_stiffness{50};
    const double rotational_stiffness{0.0};
    Eigen::MatrixXd stiffness(6, 6), damping(6, 6);
    stiffness.setZero();
    stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
    stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
    damping.setZero();
    damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) * Eigen::MatrixXd::Identity(3, 3);
    damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) * Eigen::MatrixXd::Identity(3, 3);

    try
    {
        franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
        robot.automaticErrorRecovery();
        setDefaultBehavior(robot);
        franka::Model model = robot.loadModel();
        franka::RobotState initial_state = robot.readOnce();

        // 1. Initialize Gripper Thread
        franka::Gripper gripper(argv[1]);
        gripper_thread = std::thread([&gripper]() {
            while (keep_running) {
                try {
                    franka::GripperState state = gripper.readOnce();
                    shared_gripper_width = state.width;
                } catch (...) {}
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        });

        // 2. Initialize File
        csv_file.open(argv[2], std::ios::out);
        // Header without is_grasped
        csv_file << "timestamp_us,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7,"
                 << "T01,T02,T03,T04,T05,T06,T07,T08,T09,T10,T11,T12,T13,T14,T15,T16,"
                 << "F_x,F_y,F_z,tau_x,tau_y,tau_z,tau_J1,tau_J2,tau_J3,tau_J4,tau_J5,tau_J6,tau_J7,"
                 << "gripper_width,v_x,v_y,v_z,w_x,w_y,w_z\n";

        Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
        Eigen::Vector3d position_d(initial_transform.translation());
        Eigen::Quaterniond orientation_d(initial_transform.linear());

        robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}}, 
                                   {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                                   {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}}, 
                                   {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

        auto impedance_control_callback = [&](const franka::RobotState &robot_state, franka::Duration duration) -> franka::Torques {
            // A. Logging
            auto now = std::chrono::high_resolution_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
            
            csv_file << timestamp << ",";
            for(double q : robot_state.q) csv_file << q << ",";
            for(double dq : robot_state.dq) csv_file << dq << ",";
            for(double t : robot_state.O_T_EE) csv_file << t << ",";
            for(double f : robot_state.O_F_ext_hat_K) csv_file << f << ",";
            for(double tau : robot_state.tau_ext_hat_filtered) csv_file << tau << ",";
            
            csv_file << shared_gripper_width.load() << ",";

            // B. Control Logic
            std::array<double, 7> coriolis_array = model.coriolis(robot_state);
            std::array<double, 42> jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

            Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
            Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
            
            Eigen::Matrix<double, 6, 1> twist = jacobian * dq;
            for(int i=0; i<6; i++) csv_file << twist(i) << (i==5 ? "" : ",");
            csv_file << "\n"; 

            Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
            Eigen::Vector3d position(transform.translation());
            Eigen::Quaterniond orientation(transform.linear());

            Eigen::Matrix<double, 6, 1> error;
            error.head(3) << position - position_d;
            if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) orientation.coeffs() << -orientation.coeffs();
            Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d);
            error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
            error.tail(3) << -transform.linear() * error.tail(3);

            Eigen::VectorXd tau_task(7), tau_jla(7), tau_d(7);
            tau_task << jacobian.transpose() * (-damping * (jacobian * dq)); 
            
            Eigen::MatrixXd N = Eigen::MatrixXd::Identity(7, 7) - ((Eigen::MatrixXd)jacobian.completeOrthogonalDecomposition().pseudoInverse()) * jacobian;
            Eigen::VectorXd q_dot_d = 1 * N * (q_mid - q);
            tau_jla = 4 * (q_dot_d - dq);
            tau_d << tau_task + coriolis + tau_jla;

            std::array<double, 7> tau_d_array{};
            Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;
            return tau_d_array;
        };

        std::cout << ">>> Ready! Press Enter to start control loop." << std::endl;
        std::cin.ignore();
        robot.control(impedance_control_callback);

    }
    catch (const franka::Exception &ex)
    {
        std::cout << "Exception: " << ex.what() << std::endl;
    }

    // Cleanup sequence
    keep_running = false;
    if (gripper_thread.joinable()) gripper_thread.join();
    if (csv_file.is_open()) csv_file.close();

    std::cout << ">>> Recording finished, files saved." << std::endl;
    return 0;
}