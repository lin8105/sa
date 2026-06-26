#include <iostream>
#include <string>
#include <franka/robot.h>
#include <franka/model.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <机械臂IP>" << std::endl;
        return -1;
    }
    
    try {
        // 连接机械臂
        franka::Robot robot(argv[1]);
        
        // 告诉 Python 我准备好了
        std::cout << "READY" << std::endl;

        std::string cmd;
        // 死循环监听 Python 发来的命令
        while (std::cin >> cmd) {
            if (cmd == "s") {
                // 瞬间抓取当前这一毫秒的状态
                franka::RobotState state = robot.readOnce();
                
                // 将 16 个矩阵元素以空格分隔发给 Python
                for (int i = 0; i < 16; i++) {
                    std::cout << state.O_T_EE[i] << (i == 15 ? "" : " ");
                }
                std::cout << std::endl;
            } 
            else if (cmd == "q") {
                break; // 收到退出指令，安全结束
            }
        }
    } catch (const std::exception& e) {  // 改为 std::exception
        std::cout << "ERROR " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
