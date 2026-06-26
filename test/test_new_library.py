import serial.tools.list_ports
import pyrobotiqgripper as rq
import time

# =====================================================================
# 🛠️ 终极降维打击：劫持串口扫描接口
# 无论库内部怎么调用扫描，我们都强制让它只看到 /dev/ttyUSB0
# =====================================================================
def fake_comports():
    class MockPort:
        def __init__(self, device):
            self.device = device
            self.description = "Robotiq Gripper"
            self.hwid = "USB"
    return [MockPort("/dev/ttyUSB0")]

serial.tools.list_ports.comports = fake_comports
# =====================================================================

print("正在初始化连接 (端口扫描已彻底禁用)...")

try:
    # 直接实例化，它会调用我们伪造的 comports()，直接跳过所有扫描
    gripper = rq.RobotiqGripper()
    print("【成功】串口秒连成功！")

    # 1. 激活
    gripper.activate()
    print("【成功】激活完成。")

    # 2. 毫米级校准
    gripper.calibrate_mm(0, 50)
    print("【成功】校准完成。")

    # 3. 动作控制测试
    print("正在测试精确移动到 25mm...")
    gripper.move_mm(25)
    time.sleep(1.0)
    
    print("正在测试张开...")
    gripper.open()
    time.sleep(1.0)
    
    print("正在测试闭合...")
    gripper.close()
    
    print("【完成】所有动作用例执行完毕，未产生任何冗余扫描输出。")

except Exception as e:
    print(f"运行异常: {e}")