import os
import glob
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def split_into_segments(df, max_missing_frames=30):
    """
    將整個 DataFrame 依據 'Visibility' 連續為 0 的段落切割成多個 segments。
    當連續 >= max_missing_frames 幀為不可見時，視為一段的結束。
    回傳一個 list，每個元素為一個 segment 的 DataFrame。
    """
    segments = []
    start_idx = 0
    consecutive_missing = 0
    missing_run_start = -1

    for i in range(len(df)):
        vis = df.loc[i, 'Visibility']
        if vis == 0:
            if consecutive_missing == 0:
                missing_run_start = i
            consecutive_missing += 1
        else:
            if consecutive_missing >= max_missing_frames:
                segment = df.iloc[start_idx : missing_run_start].copy()
                segments.append(segment)
                start_idx = i
            consecutive_missing = 0

    if start_idx < len(df):
        segment = df.iloc[start_idx:].copy()
        segments.append(segment)

    return segments


def smooth_positions(df, smoothing_window=5):
    df['x_smooth'] = df['X'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    df['y_smooth'] = df['Y'].rolling(window=smoothing_window, center=True, min_periods=1).mean()
    return df


def compute_speed(df):
    """
    根據平滑後的 (x_smooth, y_smooth) 計算相鄰 frame 的速度
    """
    speeds = [0.0]
    for i in range(1, len(df)):
        dx = df.loc[i, 'x_smooth'] - df.loc[i-1, 'x_smooth']
        dy = df.loc[i, 'y_smooth'] - df.loc[i-1, 'y_smooth']
        spd = math.sqrt(dx*dx + dy*dy)
        speeds.append(spd)
    return speeds


def filter_static_segments(df, speed_threshold=5.0, min_static_frames=5, smoothing_window=5, static_radius=5):
    """
    1. 先只保留 Visibility > 0 的 frame
    2. 平滑處理與計算速度，標記速度小的 frame 為 is_static
    3. 移除前後端靜止區段
    回傳: (filtered_df, original_df)
    """
    df = df[df['Visibility'] > 0].copy()
    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        return df, df

    df = smooth_positions(df, smoothing_window)
    df['speed'] = compute_speed(df)
    df['is_static'] = df['speed'] < speed_threshold

    # 前端候選區段
    front_static_count = 0
    for is_static in df['is_static']:
        if is_static:
            front_static_count += 1
        else:
            break

    new_front_count = 0
    if front_static_count > 0:
        pivot_front = df.iloc[0]
        once = True
        for i in range(front_static_count):
            dist = math.sqrt((df.loc[i, 'X'] - pivot_front['X'])**2 + (df.loc[i, 'Y'] - pivot_front['Y'])**2)
            if dist <= static_radius:
                new_front_count += 1
            else:
                if once and i < front_static_count / 2:
                    pivot_front = df.iloc[i]
                    once = False
                else:
                    break

    # 後端候選區段
    back_static_count = 0
    for is_static in reversed(df['is_static'].tolist()):
        if is_static:
            back_static_count += 1
        else:
            break

    new_back_count = 0
    if back_static_count > 0:
        pivot_back = df.iloc[len(df) - 1]
        once = True
        for i in reversed(range(len(df) - back_static_count, len(df))):
            dist = math.sqrt((df.loc[i, 'X'] - pivot_back['X'])**2 + (df.loc[i, 'Y'] - pivot_back['Y'])**2)
            if dist <= static_radius:
                new_back_count += 1
            else:
                if once and i > back_static_count / 2:
                    pivot_back = df.iloc[i]
                    once = False
                else:
                    break

    start_idx = new_front_count if front_static_count >= min_static_frames else 0
    end_idx = len(df) - new_back_count if back_static_count >= min_static_frames else len(df)

    filtered_df = df.iloc[start_idx:end_idx].copy()
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df, df


def plot_segments(original_df, filtered_df, save_path):
    """
    繪製可見軌跡（藍線）與被移除的靜止點（橘色圓點）
    原點置於左上角，即 y 軸反轉
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.title('Static Point Filtering Comparison')
    plt.xlabel('X Position (px)')
    plt.ylabel('Y Position (px)')
    ax.invert_yaxis()
    ax.grid(True)

    # 畫可見軌跡（藍線）
    plt.plot(original_df['X'], original_df['Y'],
             color='slateblue', linewidth=1.0, label='Visible Trajectory')

    # 找出被移除的靜止點（在 original 但不在 filtered）
    if 'Frame' in original_df.columns and 'Frame' in filtered_df.columns:
        removed_mask = ~original_df['Frame'].isin(filtered_df['Frame'])
    else:
        kept_indices = set(filtered_df.index)
        removed_mask = ~original_df.index.isin(kept_indices)

    removed_df = original_df[removed_mask]
    plt.scatter(removed_df['X'], removed_df['Y'],
                color='orange', s=30, zorder=5, label='Final Static (filtered)')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Static Ball Label Removal Preprocessor')
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--speed_threshold', type=float, default=10.0)
    parser.add_argument('--min_static_frames', type=int, default=5)
    parser.add_argument('--smoothing_window', type=int, default=1)
    parser.add_argument('--max_missing_frames', type=int, default=20)
    parser.add_argument('--static_radius', type=float, default=6.0)
    args = parser.parse_args()

    # ── 自動對應子資料夾 ──────────────────────────────────────────
    # 輸入：original_csv/
    input_folder        = os.path.join(args.dataset_folder, 'csv')
    # 輸出：
    before_csv_folder   = os.path.join(args.dataset_folder, 'static_removal_before_csv')
    after_csv_folder    = os.path.join(args.dataset_folder, 'static_removal_after_csv')
    static_vis_folder   = os.path.join(args.dataset_folder, 'static_vis_data_csv')
    plot_folder         = os.path.join(args.dataset_folder, 'static_removal')

    for folder in [before_csv_folder, after_csv_folder, static_vis_folder, plot_folder]:
        os.makedirs(folder, exist_ok=True)

    # ── 掃描所有 CSV ──────────────────────────────────────────────
    csv_files = glob.glob(os.path.join(input_folder, '**', '*.csv'), recursive=True) + \
                glob.glob(os.path.join(input_folder, '*.csv'))
    csv_files = list(set(csv_files))  # 去重

    if not csv_files:
        print(f"❌ 在 {input_folder} 下找不到任何 CSV 檔案")
        return

    print(f"找到 {len(csv_files)} 個 CSV 檔案，開始處理...\n")

    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        base_name = os.path.splitext(file_name)[0]

        df_all = pd.read_csv(csv_file)
        if 'Visibility' not in df_all.columns:
            print(f"⚠️  {file_name} 缺少 Visibility 欄位，跳過。")
            continue

        # 複製原始 CSV → static_removal_before_csv/
        df_all.to_csv(os.path.join(before_csv_folder, file_name), index=False)

        # Step 1: 切段
        segments = split_into_segments(df_all, max_missing_frames=args.max_missing_frames)

        # Step 2: 靜止段過濾
        filtered_segments = []
        original_segments = []

        for seg_id, seg_df in enumerate(segments):
            seg_df = seg_df.reset_index(drop=True)
            filtered_df, original_df = filter_static_segments(
                seg_df,
                speed_threshold=args.speed_threshold,
                min_static_frames=args.min_static_frames,
                smoothing_window=args.smoothing_window,
                static_radius=args.static_radius
            )
            filtered_df['segment_id'] = seg_id
            original_df['segment_id'] = seg_id
            filtered_segments.append(filtered_df)
            original_segments.append(original_df)

        final_filtered = pd.concat(filtered_segments, ignore_index=True)
        final_original = pd.concat(original_segments, ignore_index=True)

        # 輸出 after CSV → static_removal_after_csv/
        final_filtered.to_csv(os.path.join(after_csv_folder, file_name), index=False)

        # 輸出 static_vis_data_csv（保留速度、平滑欄位，方便debug）
        final_original.to_csv(os.path.join(static_vis_folder, file_name), index=False)

        # 輸出比對圖 → static_removal/
        plot_path = os.path.join(plot_folder, f'{base_name}_comparison.png')
        plot_segments(final_original, final_filtered, plot_path)

        print(f"✅ {file_name}")
        print(f"   分段數量: {len(segments)}")
        print(f"   原始可見幀: {len(final_original)} | 過濾後: {len(final_filtered)}")
        print(f"   比對圖: {plot_path}\n")

    print("全部處理完畢！")


if __name__ == "__main__":
    main()