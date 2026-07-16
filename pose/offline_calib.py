import cv2
import numpy as np
import csv
import os

# ================= 1. 从你的终端日志里抢救回来的时间戳！=================
# 键为图片编号，值为 Python 时间戳 (秒)
capture_events = {
    1: 1781534676.772,  2: 1781534683.648,  3: 1781534691.893,
    4: 1781534697.673,  5: 1781534711.995,  6: 1781534717.879,
    7: 1781534726.090,  8: 1781534733.298,  9: 1781534733.371,
    10: 1781534739.252, 11: 1781534746.296, 12: 1781534758.662,
    13: 1781534768.961, 14: 1781534784.419, 15: 1781534796.292,
    16: 1781534802.562
}

SAVE_DIR = "./calib_data"
CSV_PATH = f"{SAVE_DIR}/calib_robot_states.csv"

# 奥比中光 Femto Mega 出厂内参
camera_matrix = np.array([[1365.12, 0.0, 960.0],
                          [0.0, 1365.12, 540.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
dist_coeffs = np.array([0.012, -0.003, 0.0, 0.0, 0.0], dtype=np.float32)

marker_length = 0.05 
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def read_csv_safely(filepath):
    """防弹级 CSV 读取器：遇到长度不对的残缺行直接扔掉，绝不报错"""
    data = []
    print(f"📂 正在暴力解析 CSV 文件: {filepath} ...")
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            expected_cols = len(header)
        except StopIteration:
            return None
            
        for i, row in enumerate(reader):
            if len(row) == expected_cols:
                data.append(row)
            else:
                print(f"  🗑️ 已自动跳过第 {i+2} 行的残缺脏数据 (只有 {len(row)} 列)")
                
    return np.array(data, dtype=np.float64)

def offline_calibration():
    # 1. 安全加载 CSV 数据
    csv_data = read_csv_safely(CSV_PATH)
    if csv_data is None or len(csv_data) == 0:
        print("❌ CSV 数据为空！")
        return
        
    csv_timestamps_us = csv_data[:, 0]

    R_gripper2base, t_gripper2base = [], []
    R_marker2cam, t_marker2cam = [], []
    valid_count = 0

    print("\n⏱️ 正在将 16 张照片与机械臂底层 1000Hz 轨迹进行时间戳极速对齐...")
    
    # 2. 遍历抢救回来的 16 个时间戳
    for img_idx, py_timestamp in capture_events.items():
        img_path = f"{SAVE_DIR}/image_{img_idx}.png"
        if not os.path.exists(img_path):
            print(f"  ⚠️ 找不到图片 {img_path}，已跳过。")
            continue

        target_ts_us = int(py_timestamp * 1e6)
        
        # 寻找误差最小的那一帧
        closest_idx = np.argmin(np.abs(csv_timestamps_us - target_ts_us))
        time_diff_ms = abs(csv_timestamps_us[closest_idx] - target_ts_us) / 1000.0
        
        if time_diff_ms > 50:
            print(f"  ⚠️ image_{img_idx}.png 对齐误差过大 ({time_diff_ms:.1f}ms)，已自动剔除。")
            continue

        # 第 15 到 30 列是 O_T_EE (法兰坐标)
        O_T_EE_flat = csv_data[closest_idx, 15:31]
        T_base_gripper = O_T_EE_flat.reshape(4, 4).T 
        R_arm = T_base_gripper[:3, :3]
        t_arm = T_base_gripper[:3, 3].reshape(3, 1)

        # OpenCV 视觉提取
        img = cv2.imread(img_path)
        corners, ids, _ = detector.detectMarkers(img)
        
        if ids is not None:
            obj_points = np.array([[-marker_length/2,  marker_length/2, 0],
                                   [ marker_length/2,  marker_length/2, 0],
                                   [ marker_length/2, -marker_length/2, 0],
                                   [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(obj_points, corners[0][0], camera_matrix, dist_coeffs)
            R_cam, _ = cv2.Rodrigues(rvec)
            
            R_marker2cam.append(R_cam)
            t_marker2cam.append(tvec)
            R_gripper2base.append(R_arm)
            t_gripper2base.append(t_arm)
            valid_count += 1
            print(f"  ✅ image_{img_idx}.png 完美对齐! (误差: {time_diff_ms:.1f}ms)")
        else:
            print(f"  ⚠️ image_{img_idx}.png 未检测到 ArUco 标签 (可能被遮挡或模糊)，剔除。")

    # 3. 终极解算
    if valid_count >= 5:
        print(f"\n🎉 成功提取 {valid_count} 组有效配对，正在解算全局外参...")
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_marker2cam, t_marker2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base.flatten()

        print("\n🏆 【标定大功告成】 相机相对于机械臂基座的外参矩阵为：\n")
        print(np.round(T_cam2base, 4))
        
        np.save("T_cam2base_extrinsic.npy", T_cam2base)
        print("\n💾 矩阵已安全存为当前目录下的 'T_cam2base_extrinsic.npy'")
    else:
        print("\n❌ 能够识别到二维码的清晰照片不足 5 组，算不出矩阵，只能重新拍了。")

if __name__ == "__main__":
    offline_calibration()