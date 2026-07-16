import subprocess
import cv2
import numpy as np
import time
import os
import glob
import shutil
from pyorbbecsdk import Pipeline, Config, OBFormat, OBAlignMode, OBSensorType

# ================= 配置区 =================
CPP_EXECUTABLE = "./build/path_recorder_controller"  # 确保名字和你的C++编译输出一致
ROBOT_IP = "192.168.3.100"
SAVE_DIR = "./calibration_data"       # 数据保存主目录
OUTLIER_DIR = os.path.join(SAVE_DIR, "outliers") # 劣质数据隔离区

marker_length = 0.05 
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 误差容忍阈值（米），计算出的标定板位置如果与平均值偏差大于此值，视为劣质数据
ERROR_THRESHOLD = 0.030  # 1.5 厘米

# ================= 核心算法区 =================
def ensure_directories():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTLIER_DIR, exist_ok=True)

def get_next_index():
    """获取下一个保存文件的编号，防止覆盖"""
    existing_files = glob.glob(os.path.join(SAVE_DIR, "color_*.png"))
    if not existing_files:
        return 1
    indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in existing_files]
    return max(indices) + 1

def process_image_and_pose(bgr_image, pose_matrix, camera_matrix, dist_coeffs):
    """解析图像和位姿，返回用于标定的矩阵。处理 Eye-to-Hand 的逆变换"""
    corners, ids, _ = detector.detectMarkers(bgr_image)
    if ids is not None:
        obj_points = np.array([[-marker_length/2,  marker_length/2, 0],
                               [ marker_length/2,  marker_length/2, 0],
                               [ marker_length/2, -marker_length/2, 0],
                               [-marker_length/2, -marker_length/2, 0]], dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(obj_points, corners[0][0], camera_matrix, dist_coeffs)
        R_cam, _ = cv2.Rodrigues(rvec)
        
        # ⚠️ 注意：Eye-to-Hand 标定，需要传入 Base 到 Gripper 的变换（即 Robot Pose 的逆）
        T_base_gripper = pose_matrix
        T_gripper_base = np.linalg.inv(T_base_gripper)
        
        R_b2g = T_gripper_base[:3, :3]
        t_b2g = T_gripper_base[:3, 3].reshape(3, 1)
        
        return R_b2g, t_b2g, R_cam, tvec, True
    return None, None, None, None, False

def optimize_and_filter(data_pool):
    """
    核心优化器：计算外参 -> 评估每个点的误差 -> 剔除劣质数据 -> 重新计算
    data_pool 结构: dict { index: (R_b2g, t_b2g, R_m2c, t_m2c) }
    """
    if len(data_pool) < 5:
        print(f"⚠️ 当前有效数据仅 {len(data_pool)} 组，至少需要 5 组才能进行评估。")
        return None
    
    indices = list(data_pool.keys())
    R_b2g_list = [data_pool[i][0] for i in indices]
    t_b2g_list = [data_pool[i][1] for i in indices]
    R_m2c_list = [data_pool[i][2] for i in indices]
    t_m2c_list = [data_pool[i][3] for i in indices]
    
    # 1. 初始标定解算
    R_c2b, t_c2b = cv2.calibrateHandEye(R_b2g_list, t_b2g_list, R_m2c_list, t_m2c_list, method=cv2.CALIB_HAND_EYE_TSAI)
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_c2b
    T_cam2base[:3, 3] = t_c2b.flatten()
    
    # 2. 误差评估：对于 Eye-to-Hand，标定板相对于机械臂末端的位姿应当是一个绝对常数。
    # 我们反推每个点计算出的标定板坐标，看谁偏离了集体平均值。
    t_marker2gripper_list = []
    for R_b, t_b, R_m, t_m in zip(R_b2g_list, t_b2g_list, R_m2c_list, t_m2c_list):
        T_b2g = np.eye(4); T_b2g[:3, :3] = R_b; T_b2g[:3, 3] = t_b.flatten()
        T_m2c = np.eye(4); T_m2c[:3, :3] = R_m; T_m2c[:3, 3] = t_m.flatten()
        
        T_g2b = np.linalg.inv(T_b2g) # Robot Pose
        # T_marker2gripper = (T_gripper2base)^-1 * T_cam2base * T_marker2cam
        T_m2g = np.linalg.inv(T_g2b) @ T_cam2base @ T_m2c
        t_marker2gripper_list.append(T_m2g[:3, 3])
        
    t_mg_mean = np.mean(t_marker2gripper_list, axis=0)
    
    # 3. 寻找并隔离劣质数据
    bad_indices = []
    for i, t_mg in enumerate(t_marker2gripper_list):
        error_dist = np.linalg.norm(t_mg - t_mg_mean) # 欧氏距离误差（米）
        idx = indices[i]
        if error_dist > ERROR_THRESHOLD:
            bad_indices.append(idx)
            print(f"  ❌ 发现劣质数据 #{idx}，误差: {error_dist*1000:.1f} mm (已被隔离)")
            # 将物理文件移入回收站
            try:
                shutil.move(os.path.join(SAVE_DIR, f"color_{idx}.png"), os.path.join(OUTLIER_DIR, f"color_{idx}.png"))
                shutil.move(os.path.join(SAVE_DIR, f"pose_{idx}.txt"), os.path.join(OUTLIER_DIR, f"pose_{idx}.txt"))
            except Exception as e:
                pass
        else:
            print(f"  ✅ 数据 #{idx} 表现良好，误差: {error_dist*1000:.1f} mm")
            
    # 4. 如果有被剔除的数据，更新内存池并进行最终二次精确解算
    if bad_indices:
        for bad_idx in bad_indices:
            del data_pool[bad_idx]
        print(f"🔄 剔除劣质数据后剩余 {len(data_pool)} 组，正在重新进行高精度解算...")
        # 递归调用一次
        return optimize_and_filter(data_pool)
    
    return T_cam2base


# ================= 主程序 =================
def main():
    ensure_directories()
    print("正在初始化奥比中光相机...")
    pipeline = Pipeline()
    config = Config()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.RGB, 30)
    config.enable_stream(color_profile)
    config.set_align_mode(OBAlignMode.HW_MODE)
    pipeline.start(config)

    # 获取相机内参
    try:
        camera_param = pipeline.get_camera_param()
        intrinsics = camera_param.rgb_intrinsic
        dist = camera_param.rgb_distortion
        camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3], dtype=np.float32)
        print("✅ 成功读取相机芯片级内参！")
    except:
        camera_matrix = np.array([[1365.12, 0.0, 960.0], [0.0, 1365.12, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist_coeffs = np.array([0.012, -0.003, 0.0, 0.0, 0.0], dtype=np.float32)
        print("⚠️ 使用备用相机内参！")

    # 1. 预加载历史数据
    data_pool = {} # 存储有效数据对
    history_imgs = glob.glob(os.path.join(SAVE_DIR, "color_*.png"))
    if history_imgs:
        print(f"\n📂 发现 {len(history_imgs)} 组历史标定数据，正在加载...")
        for img_path in history_imgs:
            idx = int(os.path.basename(img_path).split('_')[1].split('.')[0])
            pose_path = os.path.join(SAVE_DIR, f"pose_{idx}.txt")
            if os.path.exists(pose_path):
                img = cv2.imread(img_path)
                pose_matrix = np.loadtxt(pose_path)
                Rb, tb, Rm, tm, success = process_image_and_pose(img, pose_matrix, camera_matrix, dist_coeffs)
                if success:
                    data_pool[idx] = (Rb, tb, Rm, tm)
        print(f"✅ 成功提取了 {len(data_pool)} 组有效历史数据！")

    next_idx = get_next_index()

    # 2. 启动 C++ 获取位姿的底层进程
    print("\n正在连接机械臂...")
    process = subprocess.Popen(
        [CPP_EXECUTABLE, ROBOT_IP],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )

    startup_msg = process.stdout.readline().strip()
    if startup_msg != "READY":
        print(f"❌ 机械臂连接失败: {startup_msg}")
        return
    print("✅ 机械臂连接成功！")
    
    print("\n================ 操作指南 ================")
    print("📸 按 [s] 键 : 拍摄并记录一组完全同步的位姿和图像")
    print("📊 按 [c] 键 : 评估当前所有数据误差，自动剔除劣质数据")
    print("🛑 按 [q] 键 : 最终解算、保存外参矩阵并退出程序")
    print("==========================================\n")

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None: continue
            color_frame = frames.get_color_frame()
            if color_frame is None: continue

            width, height = color_frame.get_width(), color_frame.get_height()
            color_image = np.asanyarray(color_frame.get_data()).reshape((height, width, 3))
            bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cv2.imshow("Hand-Eye Calib", bgr_image)
            key = cv2.waitKey(1)

            # --- 录制数据 ---
            if key == ord('s'):
                process.stdin.write("s\n")
                process.stdin.flush()
                pose_str = process.stdout.readline().strip()
                try:
                    pose_vals = [float(x) for x in pose_str.split()]
                except ValueError:
                    print("⚠️ 读取机械臂位姿失败，请重试。")
                    continue

                if len(pose_vals) == 16:
                    pose_matrix = np.array(pose_vals).reshape(4, 4).T 
                    
                    # 验证图像是否能识别
                    Rb, tb, Rm, tm, success = process_image_and_pose(bgr_image, pose_matrix, camera_matrix, dist_coeffs)
                    
                    if success:
                        # 保存到硬盘
                        cv2.imwrite(os.path.join(SAVE_DIR, f"color_{next_idx}.png"), bgr_image)
                        np.savetxt(os.path.join(SAVE_DIR, f"pose_{next_idx}.txt"), pose_matrix, fmt="%.6f")
                        
                        # 加入内存池
                        data_pool[next_idx] = (Rb, tb, Rm, tm)
                        print(f"✅ 第 {next_idx} 组数据记录成功！(已存入硬盘)")
                        next_idx += 1
                    else:
                        print("⚠️ 没看清二维码，请调整机械臂角度重拍！该点未保存。")

            # --- 中途评估优化 ---
            elif key == ord('c'):
                print("\n🔍 正在评估历史与当前的所有数据质量...")
                T_cam2base = optimize_and_filter(data_pool)
                if T_cam2base is not None:
                    print("🌟 当前最优外参矩阵 (T_cam2base):\n", np.round(T_cam2base, 4))
                print("-" * 40)

            # --- 退出并保存 ---
            elif key == ord('q'):
                process.stdin.write("q\n")
                process.stdin.flush()
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ================= 最终解算与收尾 =================
    print("\n🎉 采集结束，执行最终清洗与解算...")
    T_cam2base = optimize_and_filter(data_pool)
    if T_cam2base is not None:
        print("\n🏆 标定大功告成！最终精确外参矩阵为：\n", np.round(T_cam2base, 4))
        np.save("T_cam2base_extrinsic.npy", T_cam2base)
        print("💾 已保存为 T_cam2base_extrinsic.npy")
    else:
        print("\n❌ 有效数据不足，无法完成解算。")

if __name__ == "__main__":
    main()