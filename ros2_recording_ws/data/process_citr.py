import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_and_plot_citr(bota_csv, robot_csv, output_csv='citr_features.csv', output_npy='citr_matrices.npy', output_img='citr_fingerprint.png'):
    print("🔄 正在加载数据源...")
    if not os.path.exists(bota_csv) or not os.path.exists(robot_csv):
        print(f"❌ 错误：请确保文件 {bota_csv} 和 {robot_csv} 存在。")
        return

    # 1. 严格保留原版步骤：读取两个独立的数据源
    df_bota = pd.read_csv(bota_csv)
    df_robot = pd.read_csv(robot_csv)

    # 确保时间戳升序排列（merge_asof 的硬性要求）
    df_bota = df_bota.sort_values('timestamp_us')
    df_robot = df_robot.sort_values('timestamp_us')

    print("⏱️ 正在以 Bota 100Hz 为基准进行最近邻时间对齐...")
    
    # 2. 【核心修复点】添加 suffixes 参数防止同名列互相覆盖导致真正的 'F_x' 名字消失
    # 左表(bota)重复列保持原样，右表(robot)重复列加上 '_robot' 后缀
    df_aligned = pd.merge_asof(df_bota, df_robot, on='timestamp_us', direction='nearest', suffixes=('', '_robot'))

    print("🧮 正在利用向量化高效计算 10 维 CITR 特征...")
    # 3. 此时列名绝对干净，严格从各自的源数据列中无冲突提取向量 (N x 3)
    f = df_aligned[['F_x', 'F_y', 'F_z']].values
    tau = df_aligned[['tau_x', 'tau_y', 'tau_z']].values
    v = df_aligned[['v_x', 'v_y', 'v_z']].values
    w = df_aligned[['w_x', 'w_y', 'w_z']].values

    # 4. 向量化极速计算 10 个独立内积特征值
    ff = np.sum(f * f, axis=1)
    ftau = np.sum(f * tau, axis=1)
    tautau = np.sum(tau * tau, axis=1)
    fv = np.sum(f * v, axis=1)
    tauv = np.sum(tau * v, axis=1)
    vv = np.sum(v * v, axis=1)
    fw = np.sum(f * w, axis=1)
    tauw = np.sum(tau * w, axis=1)
    vw = np.sum(v * w, axis=1)
    ww = np.sum(w * w, axis=1)

    # 5. 组装并保存 10 维特征到 CSV
    df_citr = pd.DataFrame({
        'timestamp_us': df_aligned['timestamp_us'],
        'citr_ff': ff, 'citr_ftau': ftau, 'citr_tautau': tautau,
        'citr_fv': fv, 'citr_tauv': tauv, 'citr_vv': vv,
        'citr_fw': fw, 'citr_tauw': tauw, 'citr_vw': vw, 'citr_ww': ww
    })
    df_citr.to_csv(output_csv, index=False)
    print(f"💾 CSV 10维特征表已成功落盘 -> {output_csv}")

    print("📦 正在构建 4x4 对称 CITR 连续矩阵块...")
    # 6. 构建符合 path_recorder_controller 结构的 N x 4 x 4 矩阵并存为 NPY
    N = len(df_aligned)
    citr_matrices = np.zeros((N, 4, 4))
    
    # 严格按照 C++ 的行优先铺平逻辑填充对称矩阵块
    citr_matrices[:, 0, 0], citr_matrices[:, 0, 1], citr_matrices[:, 0, 2], citr_matrices[:, 0, 3] = ff, ftau, fv, fw
    citr_matrices[:, 1, 0], citr_matrices[:, 1, 1], citr_matrices[:, 1, 2], citr_matrices[:, 1, 3] = ftau, tautau, tauv, tauw
    citr_matrices[:, 2, 0], citr_matrices[:, 2, 1], citr_matrices[:, 2, 2], citr_matrices[:, 2, 3] = fv, tauv, vv, vw
    citr_matrices[:, 3, 0], citr_matrices[:, 3, 1], citr_matrices[:, 3, 2], citr_matrices[:, 3, 3] = fw, tauw, vw, ww

    np.save(output_npy, citr_matrices)
    print(f"💾 Numpy 3D矩阵文件(维度: {citr_matrices.shape})已成功落盘 -> {output_npy}")

    print("🎨 正在生成行无偏缩放 Task Fingerprint 热力图...")
    # 7. 绘图部分：按行无偏缩放
    citr_cols = [
        'citr_ff', 'citr_ftau', 'citr_tautau',
        'citr_fv', 'citr_tauv', 'citr_vv',
        'citr_fw', 'citr_tauw', 'citr_vw', 'citr_ww'
    ]
    citr_data = df_citr[citr_cols].values.T  # 10 x N

    # 无偏缩放归一化到 [-1, 1]
    max_abs_vals = np.max(np.abs(citr_data), axis=1, keepdims=True)
    max_abs_vals[max_abs_vals == 0] = 1e-6
    citr_normalized = citr_data / max_abs_vals

    plt.figure(figsize=(10, 6))
    sns.heatmap(citr_normalized, cmap='jet', center=0, vmin=-1, vmax=1, cbar=True)

    feature_labels = [
        r'$\langle f, f \rangle$', r'$\langle f, \tau \rangle$', r'$\langle \tau, \tau \rangle$',
        r'$\langle f, v \rangle$', r'$\langle \tau, v \rangle$', r'$\langle v, v \rangle$',
        r'$\langle f, \omega \rangle$', r'$\langle \tau, \omega \rangle$', r'$\langle v, \omega \rangle$', r'$\langle \omega, \omega \rangle$'
    ]
    plt.yticks(ticks=np.arange(10) + 0.5, labels=feature_labels, rotation=0, fontsize=12)
    plt.xticks([])  
    plt.xlabel(r'$\rightarrow t$', fontsize=16, loc='left')
    
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"🎉 任务指纹可视化图表已安全保存 -> {output_img}")

if __name__ == "__main__":
    # 维持跨源对齐架构不变
    process_and_plot_citr(bota_csv='./bota_100hz.csv', robot_csv='./robot_states.csv')