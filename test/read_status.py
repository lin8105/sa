from pymodbus.client import ModbusSerialClient

client = ModbusSerialClient(port='/dev/ttyUSB0', baudrate=115200, timeout=1)

if client.connect():
    # 读取输入寄存器 0x07D0 (Status Register)
    result = client.read_input_registers(0x07D0, 1)
    if not result.isError():
        status = result.registers[0]
        # 解析状态位 (具体参考手册的位定义)
        print(f"当前夹爪状态码: {bin(status)}")
    client.close()