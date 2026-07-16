import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def process_and_plot_citr(bota_csv, robot_csv, output_csv, output_npy, output_img):
    if not os.path.exists(bota_csv) or not os.path.exists(robot_csv):
        print(f"Error: Missing input files.\n- {bota_csv}\n- {robot_csv}")
        return

    df_bota = pd.read_csv(bota_csv)
    df_robot = pd.read_csv(robot_csv)

    df_bota = df_bota.sort_values('timestamp_us')
    df_robot = df_robot.sort_values('timestamp_us')

    print("Aligning data sources on Bota 100Hz timeline...")
    df_aligned = pd.merge_asof(df_bota, df_robot, on='timestamp_us', direction='nearest', suffixes=('', '_robot'))

    print("Calculating 10D CITR features...")
    f = df_aligned[['F_x', 'F_y', 'F_z']].values
    tau = df_aligned[['tau_x', 'tau_y', 'tau_z']].values
    
    v_col = 'v_x_robot' if 'v_x_robot' in df_aligned.columns else 'v_x'
    w_col = 'w_x_robot' if 'w_x_robot' in df_aligned.columns else 'w_x'
    
    v = df_aligned[[v_col, v_col.replace('x', 'y'), v_col.replace('x', 'z')]].values
    w = df_aligned[[w_col, w_col.replace('x', 'y'), w_col.replace('x', 'z')]].values

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

    df_citr = pd.DataFrame({
        'timestamp_us': df_aligned['timestamp_us'],
        'citr_ff': ff, 'citr_ftau': ftau, 'citr_tautau': tautau,
        'citr_fv': fv, 'citr_tauv': tauv, 'citr_vv': vv,
        'citr_fw': fw, 'citr_tauw': tauw, 'citr_vw': vw, 'citr_ww': ww
    })
    df_citr.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

    N = len(df_aligned)
    citr_matrices = np.zeros((N, 4, 4))
    citr_matrices[:, 0, 0], citr_matrices[:, 0, 1], citr_matrices[:, 0, 2], citr_matrices[:, 0, 3] = ff, ftau, fv, fw
    citr_matrices[:, 1, 0], citr_matrices[:, 1, 1], citr_matrices[:, 1, 2], citr_matrices[:, 1, 3] = ftau, tautau, tauv, tauw
    citr_matrices[:, 2, 0], citr_matrices[:, 2, 1], citr_matrices[:, 2, 2], citr_matrices[:, 2, 3] = fv, tauv, vv, vw
    citr_matrices[:, 3, 0], citr_matrices[:, 3, 1], citr_matrices[:, 3, 2], citr_matrices[:, 3, 3] = fw, tauw, vw, ww

    np.save(output_npy, citr_matrices)
    print(f"Saved {output_npy}")

    citr_cols = ['citr_ff', 'citr_ftau', 'citr_tautau', 'citr_fv', 'citr_tauv', 'citr_vv', 'citr_fw', 'citr_tauw', 'citr_vw', 'citr_ww']
    citr_data = df_citr[citr_cols].values.T 

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
    print(f"Saved {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CITR Processor")
    parser.add_argument('--bota', type=str, default='data/bota_100hz.csv')
    parser.add_argument('--robot', type=str, default='data/robot_states.csv')
    parser.add_argument('--out_csv', type=str, default='data/citr_features.csv')
    parser.add_argument('--out_npy', type=str, default='data/citr_matrices.npy')
    parser.add_argument('--out_img', type=str, default='data/citr_fingerprint.png')
    
    args = parser.parse_args()
    
    process_and_plot_citr(
        bota_csv=args.bota, 
        robot_csv=args.robot, 
        output_csv=args.out_csv, 
        output_npy=args.out_npy, 
        output_img=args.out_img
    )
