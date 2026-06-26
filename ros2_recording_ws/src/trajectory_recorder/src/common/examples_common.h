// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <array>

#include <Eigen/Core>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

/**
 * @file examples_common.h
 * Contains common types and functions for the examples.
 */

/**
 * Sets a default collision behavior碰撞, joint impedance阻抗刚度, Cartesian impedance平移刚度旋转刚度, and filter frequency.
 *
 * @param[in] robot Robot instance to set behavior on.
 */
void setDefaultBehavior(franka::Robot& robot);

/**轨迹生成
 * An example showing how to generate a joint pose motion to a goal position. Adapted from:
 * Wisama Khalil and Etienne Dombre. 2002. Modeling, Identification and Control of Robots
 * (Kogan Page Science Paper edition).
 */
class MotionGenerator {
 public:
  /**
   * Creates a new MotionGenerator instance实例 for a target q.
   *
   * @param[in] speed_factor General speed factor in range [0, 1].
   * @param[in] q_goal Target joint positions.
   */
  MotionGenerator(double speed_factor, const std::array<double, 7> q_goal);

  /**
   * Sends joint position calculations
   *
   * @param[in] robot_state Current state of the robot.
   * @param[in] period Duration时长 of execution执行.
   *
   * @return Joint positions for use inside a control loop.
   * 这是直接被 libfranka 控制循环（例如 robot.control(...)）持续高频调用的核心接口。
   * 它接收当前机器人的状态 robot_state 和执行周期 period，并输出需要执行的下步位置。  
   * 随着周期时间 period 转化为秒后被累加到总时间 time_ 上，如果发现这是控制循环的第一帧（即 time_ == 0.0），
   * 程序会抓取当前机器人的实际关节位置作为起始点 (q_start_)，计算出目标位置与起始点的差值 (delta_q_)，并调用内部函数 calculateSynchronizedValues() 来规划整段轨迹。  
   * 接着，调用内部函数 calculateDesiredValues() 获取在当前时间点 time_ 机械臂 7 个关节应该增加的位移增量 (delta_q_d)，并检查是否整个运动已经完成。  
   * 最后，将起始位置 (q_start_) 加上算出的增量，封装成 franka::JointPositions 类型，并将 motion_finished 标志一同返回给底层控制器去执行真正的电机指令。
   */
  franka::JointPositions operator()(const franka::RobotState& robot_state, franka::Duration period);

 private:
  using Vector7d = Eigen::Matrix<double, 7, 1, Eigen::ColMajor>;
  using Vector7i = Eigen::Matrix<int, 7, 1, Eigen::ColMajor>;

  bool calculateDesiredValues(double t, Vector7d* delta_q_d) const;// 计算期望值
  /*该函数的作用是，针对当前给定的时间 t，基于规划好的运动曲线，严格计算出 7 个关节当前的具体位置增量。  
  它利用循环遍历 7 个关节，如果某关节本身要移动的距离比一个极小阈值常量（kDeltaQMotionFinished，值为 1e-6）还要小，就直接视为该关节无需移动且标记为完成。  
  对于需要移动的关节，算法根据当前时间 t 落在哪一个阶段来使用不同的数学插值公式：加速阶段（t < t_1_sync_）：使用一种三次多项式（包含 std::pow(t, 3.0)）平滑地提升速度和位置。  
  匀速阶段（t_1_sync_ 到 t_2_sync_）：采用线性方程（一次方），以规划好的最高同步速度（dq_max_sync_）匀速前进。  
  减速阶段（t_2_sync_ 到 t_f_sync_）：同样使用更为复杂的三次多项式组合进行平滑刹车插值计算。  
  运动结束（t >= t_f_sync_）：输出目标增量，并将该关节对应的 joint_motion_finished 标记置为真。 */

  void calculateSynchronizedValues();// 计算同步值 同步到达
  /* 首先，它初步计算 7 个关节在各自允许的最大速度和加速度下分别所需的理论最快运动总时间 (t_f)。
  如果运动距离过短以至于达不到最大速度，也会利用一个阈值公式重算出它能达到的实际最大速度 (dq_max_reach)。  
  随后，从这 7 个总时间中，提取出耗时最长的那一个，定义为全局的最大运动时间（max_t_f）。  
  最终，函数利用这个最大总时间强行约束所有关节。
  它根据二次方程求解公式反推并降低其余耗时较短的关节的同步运行速度（dq_max_sync_），
  并准确地划分出每个关节的三个关键时间节点：加速结束的时间点（t_1_sync_）、减速开始的时间点（t_2_sync_）和整个运动结束的时间点（t_f_sync_）。  */

  static constexpr double kDeltaQMotionFinished = 1e-6;
  const Vector7d q_goal_;

  Vector7d q_start_;
  Vector7d delta_q_;

  Vector7d dq_max_sync_;
  Vector7d t_1_sync_;
  Vector7d t_2_sync_;
  Vector7d t_f_sync_;
  Vector7d q_1_;

  double time_ = 0.0;

  Vector7d dq_max_ = (Vector7d() << 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5).finished();
  Vector7d ddq_max_start_ = (Vector7d() << 5, 5, 5, 5, 5, 5, 5).finished();
  Vector7d ddq_max_goal_ = (Vector7d() << 5, 5, 5, 5, 5, 5, 5).finished();
};
