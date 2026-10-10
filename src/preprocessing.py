import numpy as np
import pandas as pd


def preprocess_session(
    file_path,
    sheet_name,
    activity,
    session_id,
    grid_ms=80,
    window_size=75,
    max_ppg_gap_ms=240,
    max_acc_gap_ms=160,
    min_chest_events=6
):
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Separate sensor streams
    ppg_df = df[df["PPG_Raw"].notna()][
        ["Timestamp_ms", "PPG_Raw"]
    ].copy()

    acc_df = df[df["ACC_X"].notna()][
        ["Timestamp_ms", "ACC_X", "ACC_Y", "ACC_Z"]
    ].copy()

    chest_df = df[df["Chest_HR_BPM"].notna()][
        ["Timestamp_ms", "Chest_HR_BPM", "Chest_RR_ms"]
    ].copy()

    # Common time interval
    start_ts = max(
        ppg_df["Timestamp_ms"].iloc[0],
        acc_df["Timestamp_ms"].iloc[0]
    )

    end_ts = min(
        ppg_df["Timestamp_ms"].iloc[-1],
        acc_df["Timestamp_ms"].iloc[-1],
        chest_df["Timestamp_ms"].iloc[-1]
    )

    # 12.5 Hz common time grid
    time_grid = np.arange(
        start_ts,
        end_ts + 1,
        grid_ms
    )

    # -------------------------
    # PPG interpolation
    # -------------------------

    ppg_times = ppg_df["Timestamp_ms"].to_numpy()
    ppg_values = ppg_df["PPG_Raw"].to_numpy()

    right_idx = np.searchsorted(ppg_times, time_grid)

    ppg_resampled = np.full(len(time_grid), np.nan)

    for i, t in enumerate(time_grid):

        r = right_idx[i]

        if r < len(ppg_times) and ppg_times[r] == t:
            ppg_resampled[i] = ppg_values[r]

        elif 0 < r < len(ppg_times):

            left = r - 1

            t0 = ppg_times[left]
            t1 = ppg_times[r]

            v0 = ppg_values[left]
            v1 = ppg_values[r]

            gap = t1 - t0

            if gap <= max_ppg_gap_ms:

                ratio = (t - t0) / gap

                ppg_resampled[i] = (
                    v0 + ratio * (v1 - v0)
                )

    # -------------------------
    # ACC interpolation
    # -------------------------

    acc_times = acc_df["Timestamp_ms"].to_numpy()

    acc_values = acc_df[
        ["ACC_X", "ACC_Y", "ACC_Z"]
    ].to_numpy()

    right_idx = np.searchsorted(acc_times, time_grid)

    acc_resampled = np.full(
        (len(time_grid), 3),
        np.nan
    )

    for i, t in enumerate(time_grid):

        r = right_idx[i]

        if r < len(acc_times) and acc_times[r] == t:
            acc_resampled[i] = acc_values[r]

        elif 0 < r < len(acc_times):

            left = r - 1

            t0 = acc_times[left]
            t1 = acc_times[r]

            v0 = acc_values[left]
            v1 = acc_values[r]

            gap = t1 - t0

            if gap <= max_acc_gap_ms:

                ratio = (t - t0) / gap

                acc_resampled[i] = (
                    v0 + ratio * (v1 - v0)
                )

    # Synchronized dataframe
    sync_df = pd.DataFrame({
        "Timestamp_ms": time_grid,
        "PPG_Raw": ppg_resampled,
        "ACC_X": acc_resampled[:, 0],
        "ACC_Y": acc_resampled[:, 1],
        "ACC_Z": acc_resampled[:, 2]
    })

    # -------------------------
    # 6-second windows
    # -------------------------

    X_ppg = []
    X_fusion = []
    y = []
    metadata = []

    total_windows = len(sync_df) // window_size

    for i in range(total_windows):

        start = i * window_size
        end = start + window_size

        window = sync_df.iloc[start:end]

        start_time = window["Timestamp_ms"].iloc[0]

        end_time = start_time + (
            window_size * grid_ms
        )

        chest_window = chest_df[
            (chest_df["Timestamp_ms"] >= start_time) &
            (chest_df["Timestamp_ms"] < end_time)
        ]

        has_missing = window[
            ["PPG_Raw", "ACC_X", "ACC_Y", "ACC_Z"]
        ].isna().any().any()

        enough_chest = (
            len(chest_window) >= min_chest_events
        )

        if (not has_missing) and enough_chest:

            ppg_input = window[
                ["PPG_Raw"]
            ].to_numpy()

            fusion_input = window[
                ["PPG_Raw", "ACC_X", "ACC_Y", "ACC_Z"]
            ].to_numpy()

            target_hr = (
                chest_window["Chest_HR_BPM"].mean()
            )

            X_ppg.append(ppg_input)
            X_fusion.append(fusion_input)
            y.append(target_hr)

            metadata.append({
                "Activity": activity,
                "Session_ID": session_id,
                "Window_ID": i,
                "Start_ms": start_time,
                "Chest_events": len(chest_window),
                "Target_HR": target_hr
            })

    X_ppg = np.array(X_ppg)
    X_fusion = np.array(X_fusion)
    y = np.array(y)

    metadata = pd.DataFrame(metadata)

    print(f"\n{session_id}")
    print("--------------------------")
    print("Total windows:", total_windows)
    print("Valid windows:", len(y))
    print("Rejected:", total_windows - len(y))
    print("PPG shape:", X_ppg.shape)
    print("Fusion shape:", X_fusion.shape)
    print("Target shape:", y.shape)

    return X_ppg, X_fusion, y, metadata