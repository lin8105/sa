import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRIPPER_MAX_POSITION = 0.0247059
GRIPPER_MIN_POSITION = 0.000490196
TARGET_RATE_HZ = 100.0


def require_columns(df, required_columns, file_description):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{file_description} is missing required columns: {', '.join(missing)}"
        )


def read_csv_checked(path, file_description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{file_description} not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{file_description} is empty: {path}")
    return df


def align_gripper_to_target_timestamps(
    df_gripper,
    target_timestamps,
    method="linear",
):
    """
    Resample the gripper position to the target timeline.

    Preferred mode:
        Use timestamp_us from gripper_10hz.csv.

    Fallback mode:
        If timestamp_us is unavailable, assume uniformly sampled gripper data
        spanning the complete target time range.
    """
    require_columns(df_gripper, ["position"], "Gripper CSV")

    target_timestamps = np.asarray(target_timestamps, dtype=np.int64)
    if target_timestamps.size == 0:
        return np.array([], dtype=np.float64), "empty-target"

    positions = pd.to_numeric(
        df_gripper["position"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    valid_position_mask = np.isfinite(positions)
    if not np.any(valid_position_mask):
        raise ValueError("Gripper CSV contains no valid position values.")

    has_timestamp = (
        "timestamp_us" in df_gripper.columns
        and pd.to_numeric(df_gripper["timestamp_us"], errors="coerce")
        .notna()
        .any()
    )

    if has_timestamp:
        source = pd.DataFrame(
            {
                "timestamp_us": pd.to_numeric(
                    df_gripper["timestamp_us"], errors="coerce"
                ),
                "position": pd.to_numeric(
                    df_gripper["position"], errors="coerce"
                ),
            }
        )
        source = (
            source.dropna(subset=["timestamp_us", "position"])
            .sort_values("timestamp_us")
            .drop_duplicates("timestamp_us", keep="last")
        )

        if source.empty:
            raise ValueError(
                "Gripper CSV has a timestamp_us column, but no valid timestamp-position pairs."
            )

        source_timestamps = source["timestamp_us"].to_numpy(dtype=np.int64)
        source_positions = source["position"].to_numpy(dtype=np.float64)
        alignment_mode = "timestamp"
    else:
        source_positions = positions[valid_position_mask]

        if source_positions.size == 1:
            source_timestamps = np.array(
                [target_timestamps[0]], dtype=np.int64
            )
        else:
            source_timestamps = np.linspace(
                target_timestamps[0],
                target_timestamps[-1],
                num=source_positions.size,
            ).astype(np.int64)

        alignment_mode = "uniform-assumption"

    if source_positions.size == 1:
        aligned = np.full(
            target_timestamps.shape,
            source_positions[0],
            dtype=np.float64,
        )
        return aligned, alignment_mode

    # Subtract a common origin before interpolation. This avoids using very
    # large epoch timestamps directly in floating-point interpolation.
    time_origin = min(
        int(source_timestamps[0]),
        int(target_timestamps[0]),
    )
    source_time_relative = (
        source_timestamps.astype(np.float64) - time_origin
    )
    target_time_relative = (
        target_timestamps.astype(np.float64) - time_origin
    )

    method = method.lower()

    if method == "linear":
        aligned = np.interp(
            target_time_relative,
            source_time_relative,
            source_positions,
            left=source_positions[0],
            right=source_positions[-1],
        )

    elif method in {"zoh", "ffill", "forward_fill"}:
        source_indices = (
            np.searchsorted(
                source_time_relative,
                target_time_relative,
                side="right",
            )
            - 1
        )
        source_indices = np.clip(
            source_indices,
            0,
            source_positions.size - 1,
        )
        aligned = source_positions[source_indices]

    else:
        raise ValueError(
            f"Unsupported gripper resampling method: {method}"
        )

    return aligned, alignment_mode


def normalize_gripper_position(position_values):
    """
    Reverse normalization requested by the experiment:

    position = 0.0247059   -> 0.0
    position = 0.000490196 -> 1.0

    With the shared jet scale from -1 to 1:
    0.0 appears green and 1.0 appears red.
    """
    denominator = (
        GRIPPER_MAX_POSITION - GRIPPER_MIN_POSITION
    )
    normalized = (
        GRIPPER_MAX_POSITION
        - np.asarray(position_values, dtype=np.float64)
    ) / denominator

    return np.clip(normalized, 0.0, 1.0)


def calculate_citr_features(df_aligned):
    force = df_aligned[["F_x", "F_y", "F_z"]].to_numpy(
        dtype=np.float64
    )
    torque = df_aligned[
        ["tau_x", "tau_y", "tau_z"]
    ].to_numpy(dtype=np.float64)

    velocity_columns = (
        ["v_x_robot", "v_y_robot", "v_z_robot"]
        if all(
            column in df_aligned.columns
            for column in ["v_x_robot", "v_y_robot", "v_z_robot"]
        )
        else ["v_x", "v_y", "v_z"]
    )

    angular_velocity_columns = (
        ["w_x_robot", "w_y_robot", "w_z_robot"]
        if all(
            column in df_aligned.columns
            for column in ["w_x_robot", "w_y_robot", "w_z_robot"]
        )
        else ["w_x", "w_y", "w_z"]
    )

    require_columns(
        df_aligned,
        velocity_columns + angular_velocity_columns,
        "Aligned robot data",
    )

    velocity = df_aligned[velocity_columns].to_numpy(
        dtype=np.float64
    )
    angular_velocity = df_aligned[
        angular_velocity_columns
    ].to_numpy(dtype=np.float64)

    ff = np.sum(force * force, axis=1)
    ftau = np.sum(force * torque, axis=1)
    tautau = np.sum(torque * torque, axis=1)
    fv = np.sum(force * velocity, axis=1)
    tauv = np.sum(torque * velocity, axis=1)
    vv = np.sum(velocity * velocity, axis=1)
    fw = np.sum(force * angular_velocity, axis=1)
    tauw = np.sum(torque * angular_velocity, axis=1)
    vw = np.sum(velocity * angular_velocity, axis=1)
    ww = np.sum(
        angular_velocity * angular_velocity,
        axis=1,
    )

    return {
        "citr_ff": ff,
        "citr_ftau": ftau,
        "citr_tautau": tautau,
        "citr_fv": fv,
        "citr_tauv": tauv,
        "citr_vv": vv,
        "citr_fw": fw,
        "citr_tauw": tauw,
        "citr_vw": vw,
        "citr_ww": ww,
    }


def normalize_citr_rows(citr_data):
    """
    Normalize each CITR feature independently to [-1, 1].

    citr_data shape:
        number_of_features x number_of_samples
    """
    max_absolute_values = np.max(
        np.abs(citr_data),
        axis=1,
        keepdims=True,
    )
    max_absolute_values[
        max_absolute_values == 0
    ] = 1.0

    return np.clip(
        citr_data / max_absolute_values,
        -1.0,
        1.0,
    )


def build_citr_matrices(feature_values):
    number_of_samples = len(feature_values["citr_ff"])
    matrices = np.zeros(
        (number_of_samples, 4, 4),
        dtype=np.float64,
    )

    ff = feature_values["citr_ff"]
    ftau = feature_values["citr_ftau"]
    tautau = feature_values["citr_tautau"]
    fv = feature_values["citr_fv"]
    tauv = feature_values["citr_tauv"]
    vv = feature_values["citr_vv"]
    fw = feature_values["citr_fw"]
    tauw = feature_values["citr_tauw"]
    vw = feature_values["citr_vw"]
    ww = feature_values["citr_ww"]

    matrices[:, 0, 0] = ff
    matrices[:, 0, 1] = ftau
    matrices[:, 0, 2] = fv
    matrices[:, 0, 3] = fw

    matrices[:, 1, 0] = ftau
    matrices[:, 1, 1] = tautau
    matrices[:, 1, 2] = tauv
    matrices[:, 1, 3] = tauw

    matrices[:, 2, 0] = fv
    matrices[:, 2, 1] = tauv
    matrices[:, 2, 2] = vv
    matrices[:, 2, 3] = vw

    matrices[:, 3, 0] = fw
    matrices[:, 3, 1] = tauw
    matrices[:, 3, 2] = vw
    matrices[:, 3, 3] = ww

    return matrices


def save_heatmap_column_mapping(
    output_map,
    timestamps,
    gripper_position,
    gripper_normalized,
):
    sample_indices = np.arange(
        len(timestamps),
        dtype=np.int64,
    )

    mapping = pd.DataFrame(
        {
            "heatmap_column": sample_indices,
            "sample_index_100hz": sample_indices,
            "timestamp_us": np.asarray(
                timestamps,
                dtype=np.int64,
            ),
            "relative_time_s": (
                np.asarray(timestamps, dtype=np.int64)
                - int(timestamps[0])
            )
            / 1_000_000.0,
            "gripper_position": np.asarray(
                gripper_position,
                dtype=np.float64,
            ),
            "gripper_norm": np.asarray(
                gripper_normalized,
                dtype=np.float64,
            ),
        }
    )

    mapping.to_csv(output_map, index=False)
    print(f"Saved {output_map}")


def calculate_figure_width(
    sample_count,
    target_rate_hz,
    inches_per_second,
    minimum_width,
    maximum_width,
):
    duration_seconds = sample_count / target_rate_hz
    width = max(
        minimum_width,
        duration_seconds * inches_per_second,
    )

    if maximum_width > 0:
        width = min(width, maximum_width)

    return width


def save_pure_heatmap(
    combined_data,
    output_pure_img,
    row_height_pixels=8,
):
    """
    Save a pure RGB heatmap without title, axes, ticks, labels, margins,
    or colorbar.

    Important properties:
    - output width is exactly the number of trajectory samples
    - image column t corresponds exactly to sample/frame t
    - only the vertical direction is enlarged
    - nearest-neighbor row repetition avoids interpolation between
      adjacent time columns

    combined_data shape:
        number_of_rows x number_of_samples
    """
    if combined_data.ndim != 2:
        raise ValueError(
            "combined_data must have shape [rows, samples]."
        )

    if row_height_pixels < 1:
        raise ValueError(
            "row_height_pixels must be at least 1."
        )

    # Repeat only along the feature-row axis. The time-axis width remains
    # unchanged, so PNG column t maps directly to trajectory sample t.
    enlarged_data = np.repeat(
        combined_data,
        repeats=row_height_pixels,
        axis=0,
    )

    # Convert the fixed [-1, 1] scale to RGB using the same jet colormap
    # as the labeled visualization.
    normalized_for_colormap = np.clip(
        (enlarged_data + 1.0) / 2.0,
        0.0,
        1.0,
    )
    rgb_image = plt.get_cmap("jet")(
        normalized_for_colormap
    )[..., :3]

    plt.imsave(
        output_pure_img,
        rgb_image,
        origin="upper",
    )

    print(
        f"Saved {output_pure_img} "
        f"(pure heatmap, shape="
        f"{rgb_image.shape[0]}x{rgb_image.shape[1]}x3)"
    )


def save_combined_heatmap(
    combined_data,
    output_img,
    target_rate_hz,
    inches_per_second,
    minimum_figure_width,
    maximum_figure_width,
    dpi,
):
    """
    Draw one single 11 x N heatmap.

    The first ten rows and the gripper row share:
    - exactly the same x coordinates
    - exactly the same number of columns
    - exactly the same colormap
    - exactly the same value scale [-1, 1]

    Figure width grows with recording duration. Therefore the horizontal
    scale remains constant across recordings.
    """
    sample_count = combined_data.shape[1]

    figure_width = calculate_figure_width(
        sample_count=sample_count,
        target_rate_hz=target_rate_hz,
        inches_per_second=inches_per_second,
        minimum_width=minimum_figure_width,
        maximum_width=maximum_figure_width,
    )

    figure_height = 6.6

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=dpi,
    )

    image = axis.imshow(
        combined_data,
        cmap="jet",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[
            0,
            sample_count,
            combined_data.shape[0],
            0,
        ],
    )

    row_labels = [
        r"$\langle f,f\rangle$",
        r"$\langle f,\tau\rangle$",
        r"$\langle \tau,\tau\rangle$",
        r"$\langle f,v\rangle$",
        r"$\langle \tau,v\rangle$",
        r"$\langle v,v\rangle$",
        r"$\langle f,\omega\rangle$",
        r"$\langle \tau,\omega\rangle$",
        r"$\langle v,\omega\rangle$",
        r"$\langle \omega,\omega\rangle$",
        "gripper width",
    ]

    axis.set_yticks(
        np.arange(len(row_labels)) + 0.5
    )
    axis.set_yticklabels(
        row_labels,
        rotation=0,
        fontsize=11,
    )

    # Use a fixed tick interval in seconds so different recordings keep the
    # same horizontal time scale.
    tick_interval_seconds = 1.0
    tick_interval_samples = max(
        1,
        int(round(
            tick_interval_seconds * target_rate_hz
        )),
    )

    tick_positions = np.arange(
        0,
        sample_count + 1,
        tick_interval_samples,
        dtype=np.int64,
    )

    if tick_positions.size == 0 or tick_positions[-1] != sample_count:
        tick_positions = np.append(
            tick_positions,
            sample_count,
        )

    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [str(int(position)) for position in tick_positions],
        rotation=0,
        fontsize=9,
    )

    axis.set_xlabel(
        "100 Hz sample index / heatmap column",
        fontsize=11,
    )
    axis.set_ylabel("")

    # Draw horizontal boundaries so the gripper row is visually part of the
    # same aligned heatmap, while still being easy to identify.
    for boundary in range(1, combined_data.shape[0]):
        axis.axhline(
            boundary,
            linewidth=0.25,
            color="black",
            alpha=0.25,
        )

    axis.axhline(
        10,
        linewidth=1.2,
        color="black",
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.015,
        fraction=0.025,
    )
    colorbar.set_label(
        "Normalized value",
        rotation=90,
    )

    axis.text(
        1.01,
        (10.5 / 11.0),
        "gripper: green=open, red=closed",
        transform=axis.transAxes,
        va="center",
        ha="left",
        fontsize=9,
    )

    figure.tight_layout()
    figure.savefig(
        output_img,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(
        f"Saved {output_img} "
        f"({sample_count} columns, width={figure_width:.2f} in)"
    )


def process_and_plot_citr(
    bota_csv,
    robot_csv,
    gripper_csv,
    output_csv,
    output_npy,
    output_img,
    output_pure_img,
    output_map,
    pure_row_height_pixels=8,
    gripper_resample="linear",
    target_rate_hz=TARGET_RATE_HZ,
    inches_per_second=2.5,
    minimum_figure_width=10.0,
    maximum_figure_width=0.0,
    dpi=150,
):
    missing_files = [
        path
        for path in [
            bota_csv,
            robot_csv,
            gripper_csv,
        ]
        if not os.path.isfile(path)
    ]

    if missing_files:
        for missing_file in missing_files:
            print(
                f"Missing required input file: {missing_file}",
                file=sys.stderr,
            )
        raise FileNotFoundError(
            "One or more required input files are missing."
        )

    df_bota = read_csv_checked(
        bota_csv,
        "Bota CSV",
    )
    df_robot = read_csv_checked(
        robot_csv,
        "Robot-state CSV",
    )
    df_gripper = read_csv_checked(
        gripper_csv,
        "Gripper CSV",
    )

    require_columns(
        df_bota,
        [
            "timestamp_us",
            "F_x",
            "F_y",
            "F_z",
            "tau_x",
            "tau_y",
            "tau_z",
        ],
        "Bota CSV",
    )
    require_columns(
        df_robot,
        ["timestamp_us"],
        "Robot-state CSV",
    )
    require_columns(
        df_gripper,
        ["position"],
        "Gripper CSV",
    )

    df_bota["timestamp_us"] = pd.to_numeric(
        df_bota["timestamp_us"],
        errors="coerce",
    )
    df_robot["timestamp_us"] = pd.to_numeric(
        df_robot["timestamp_us"],
        errors="coerce",
    )

    df_bota = (
        df_bota.dropna(subset=["timestamp_us"])
        .sort_values("timestamp_us")
        .drop_duplicates("timestamp_us", keep="last")
    )
    df_robot = (
        df_robot.dropna(subset=["timestamp_us"])
        .sort_values("timestamp_us")
        .drop_duplicates("timestamp_us", keep="last")
    )

    if df_bota.empty:
        raise ValueError(
            "Bota CSV contains no valid timestamps."
        )
    if df_robot.empty:
        raise ValueError(
            "Robot-state CSV contains no valid timestamps."
        )

    df_bota["timestamp_us"] = df_bota[
        "timestamp_us"
    ].astype(np.int64)
    df_robot["timestamp_us"] = df_robot[
        "timestamp_us"
    ].astype(np.int64)

    print(
        "Aligning robot states and gripper data "
        "on the Bota 100 Hz timeline..."
    )

    df_aligned = pd.merge_asof(
        df_bota,
        df_robot,
        on="timestamp_us",
        direction="nearest",
        suffixes=("", "_robot"),
    )

    gripper_position_aligned, gripper_alignment_mode = (
        align_gripper_to_target_timestamps(
            df_gripper=df_gripper,
            target_timestamps=df_aligned[
                "timestamp_us"
            ].to_numpy(dtype=np.int64),
            method=gripper_resample,
        )
    )

    print(
        "Gripper alignment mode: "
        f"{gripper_alignment_mode}, "
        f"resampling method: {gripper_resample}"
    )

    gripper_normalized = normalize_gripper_position(
        gripper_position_aligned
    )

    print("Calculating 10D CITR features...")
    feature_values = calculate_citr_features(
        df_aligned
    )

    df_citr = pd.DataFrame(
        {
            "timestamp_us": df_aligned[
                "timestamp_us"
            ].to_numpy(dtype=np.int64),
            **feature_values,
            "gripper_position": gripper_position_aligned,
            "gripper_norm": gripper_normalized,
        }
    )

    df_citr.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

    citr_matrices = build_citr_matrices(
        feature_values
    )
    np.save(output_npy, citr_matrices)
    print(f"Saved {output_npy}")

    feature_column_names = [
        "citr_ff",
        "citr_ftau",
        "citr_tautau",
        "citr_fv",
        "citr_tauv",
        "citr_vv",
        "citr_fw",
        "citr_tauw",
        "citr_vw",
        "citr_ww",
    ]

    citr_data = df_citr[
        feature_column_names
    ].to_numpy(dtype=np.float64).T

    citr_normalized = normalize_citr_rows(
        citr_data
    )

    # One combined 11 x N matrix. Every row has exactly the same N columns.
    combined_heatmap_data = np.vstack(
        [
            citr_normalized,
            gripper_normalized[np.newaxis, :],
        ]
    )

    if output_map:
        save_heatmap_column_mapping(
            output_map=output_map,
            timestamps=df_aligned[
                "timestamp_us"
            ].to_numpy(dtype=np.int64),
            gripper_position=gripper_position_aligned,
            gripper_normalized=gripper_normalized,
        )

    save_pure_heatmap(
        combined_data=combined_heatmap_data,
        output_pure_img=output_pure_img,
        row_height_pixels=pure_row_height_pixels,
    )

    save_combined_heatmap(
        combined_data=combined_heatmap_data,
        output_img=output_img,
        target_rate_hz=target_rate_hz,
        inches_per_second=inches_per_second,
        minimum_figure_width=minimum_figure_width,
        maximum_figure_width=maximum_figure_width,
        dpi=dpi,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Align Bota, robot-state and gripper data, "
            "calculate CITR features, and generate an aligned "
            "11-row heatmap."
        )
    )

    parser.add_argument(
        "--bota",
        type=str,
        default="data/bota_100hz.csv",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="data/robot_states.csv",
    )
    parser.add_argument(
        "--gripper",
        type=str,
        default="data/gripper_10hz.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/citr_features.csv",
    )
    parser.add_argument(
        "--out_npy",
        type=str,
        default="data/citr_matrices.npy",
    )
    parser.add_argument(
        "--out_img",
        type=str,
        default="data/citr_fingerprint.png",
    )
    parser.add_argument(
        "--out_pure_img",
        type=str,
        default="data/citr_fingerprint_pure.png",
        help=(
            "Pure RGB heatmap without axes, labels, margins, or colorbar. "
            "Its width is exactly the number of trajectory samples."
        ),
    )
    parser.add_argument(
        "--out_map",
        type=str,
        default=None,
        help=(
            "Optional CSV mapping each heatmap column to timestamps. "
            "Omit this argument to skip generating the mapping file."
        ),
    )
    parser.add_argument(
        "--pure_row_height_pixels",
        type=int,
        default=8,
        help=(
            "Vertical pixel height assigned to each heatmap feature row. "
            "Only the vertical dimension is enlarged."
        ),
    )
    parser.add_argument(
        "--gripper_resample",
        type=str,
        default="linear",
        choices=[
            "linear",
            "zoh",
            "ffill",
        ],
    )
    parser.add_argument(
        "--target_rate_hz",
        type=float,
        default=TARGET_RATE_HZ,
    )
    parser.add_argument(
        "--inches_per_second",
        type=float,
        default=2.5,
        help=(
            "Horizontal image width per second. "
            "Keeping this constant gives all recordings "
            "the same time scale."
        ),
    )
    parser.add_argument(
        "--minimum_figure_width",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--maximum_figure_width",
        type=float,
        default=0.0,
        help=(
            "Maximum width in inches. "
            "Use 0 for no maximum."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
    )

    args = parser.parse_args()

    try:
        process_and_plot_citr(
            bota_csv=args.bota,
            robot_csv=args.robot,
            gripper_csv=args.gripper,
            output_csv=args.out_csv,
            output_npy=args.out_npy,
            output_img=args.out_img,
            output_pure_img=args.out_pure_img,
            output_map=args.out_map,
            pure_row_height_pixels=args.pure_row_height_pixels,
            gripper_resample=args.gripper_resample,
            target_rate_hz=args.target_rate_hz,
            inches_per_second=args.inches_per_second,
            minimum_figure_width=args.minimum_figure_width,
            maximum_figure_width=args.maximum_figure_width,
            dpi=args.dpi,
        )
    except Exception as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
