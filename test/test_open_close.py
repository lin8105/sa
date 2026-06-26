import serial
import time

print("正在通过原生物理串口直连 /dev/ttyUSB0...")

try:
    # 1. 严格按照 Robotiq 官方电气规范初始化原生串口 (115200, 8-N-1)
    # 注意：如果夹爪波特率还是 5200，请将下面的 115200 改为 5200值
    ser = serial.Serial(
        port='/dev/ttyUSB0',
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1.0
    )
    print("【成功】原生物理串口已打通，已跳过所有第三方库的端口扫描！")

    def send_raw_bytes(hex_bytes):
        """向串口发送原始字节流并清空缓冲区"""
        ser.write(hex_bytes)
        time.sleep(0.1)
        return ser.read(ser.in_waiting)

    # ---- 步骤一：通信复位（Reset） ----
    # 报文含义: [从站ID:09] [功能码:10] [地址:03E8] [寄存器数:0003] [数据字节:06] [全0数据] [CRC校验]
    print("1. 正在发送官方通信复位报文 (清除故障码)...")
    send_raw_bytes(b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x74\x30')
    time.sleep(0.5)

    # ---- 步骤二：发送激活（Activation）动作 ----
    # 报文含义: 寄存器 0x03E8 写入 0x0100 (rACT = 1) 触发自动初始化程序
    print("2. 正在发送官方激活报文 (rACT=1)...")
    send_raw_bytes(b'\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1')
    
    print("3. 正在物理校准！等待 4 秒，夹爪此时应该会开合一次...")
    time.sleep(4.0)

    # ---- 步骤三：执行开合动作（核心修复：带运动允许控制字 0x0900） ----
    
    # 4. 完全张开 (Position=0x00, Speed=0xFF, Force=0xFF)
    # 官方标准全打包控制报文 (必须包含激活维持)：
    print("4. 正在发送完全张开指令 (Position: 0)...")
    send_raw_bytes(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19')
    time.sleep(2.0)

    # 5. 完全闭合 (Position=0xFF, Speed=0xFF, Force=0xFF)
    # 官方标准全打包控制报文：
    print("5. 正在发送完全闭合指令 (Position: 255)...")
    send_raw_bytes(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x62\x99')
    time.sleep(2.0)
    
    print("【成功】所有原生控制字执行完毕，夹爪测试结束。")

except Exception as e:
    print(f"【异常】串口运行错误: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("串口已安全关闭。")
