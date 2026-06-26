#ifndef DQ_ROBOTS_FRANKAROBOT_H
#define DQ_ROBOTS_FRANKAROBOT_H
#include <dqrobotics/DQ.h>
#include <dqrobotics/robot_modeling/DQ_Kinematics.h>
#include <dqrobotics/robot_modeling/DQ_SerialManipulator.h>
// #include <dqrobotics/utils/DQ_Geometry.h>
#include <dqrobotics/robot_modeling/DQ_SerialManipulatorMDH.h>
#include <dqrobotics/utils/DQ_Constants.h>

namespace DQ_robotics
{

class FrankaRobot
{
public:
    static MatrixXd _get_mdh_matrix();
    static DQ _get_offset_base();
    static DQ _get_offset_flange();
    static std::tuple<const VectorXd, const VectorXd> _get_q_limits();
    static std::tuple<const VectorXd, const VectorXd> _get_q_dot_limits();
    static DQ_SerialManipulatorMDH kinematics();
};

}

#endif
