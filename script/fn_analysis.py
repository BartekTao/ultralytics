"""
FN Analysis Script

使用方式：
    python fn_analysis.py \
        --gt_dir /usr/src/datasets/tracknet/val_data/sportxai_rally_test/static_removal_before_csv \
        --pred_dir /usr/src/ultralytics/runs/detect/predict83/sportxai_rally_test/csv \

輸出結構：
    runs/analysis/
    └── analysis1/
        ├── fn_all.csv
        ├── fn_summary.csv
        ├── fn_category_chart.png
        └── sample_other_miss/
            └── video1_000137/       ← 每個抽樣一個資料夾（以 FN frame 命名）
                ├── 000127.jpg       ← FN 前 10 幀
                ├── 000137_FN.jpg    ← FN 幀（紅框 + _FN 標記）
                └── 000147.jpg       ← FN 後 10 幀
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import deque
from pathlib import Path


# ===================== 分類邏輯 =====================

HIT_WINDOW = 5

def classify_fn(row, gt_df, frame_idx):
    static_ball = row.get('static_ball', False)
    motion_score = row.get('motion_score', 0.0)
    in_rally = is_in_rally(gt_df, frame_idx)
    near_hit = is_near_hit(gt_df, frame_idx, window=HIT_WINDOW)

    if static_ball:
        return 'rally中靜止球' if in_rally else '正常靜止球'
    elif near_hit:
        return '接球瞬間'
    elif motion_score > 50:
        return '高速運動模糊'
    else:
        return '其他漏偵測'


def is_in_rally(gt_df, frame_idx):
    events = gt_df[gt_df['Event'].isin([1, 2, 3])][['Frame', 'Event']].sort_values('Frame')
    if len(events) == 0:
        return False
    prev_events = events[events['Frame'] <= frame_idx]
    if len(prev_events) == 0:
        return False
    return prev_events.iloc[-1]['Event'] in [1, 3]


def is_near_hit(gt_df, frame_idx, window=5):
    hit_frames = gt_df[gt_df['hit'] == 1]['Frame'].values
    return any(abs(frame_idx - hf) <= window for hf in hit_frames)


# ===================== FN 分析 =====================

CONF_THRESHOLD = 0.5   # 與 val.py 一致
TOLERANCE_640 = 5.0    # val.py 的 tolerance5，640×640 解析度下 5 pixel

def get_video_resolution(video_path):
    """取得影片解析度"""
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def analyze_video(gt_path, pred_path, video_path):
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path)

    # 取得影片解析度，換算 tolerance 到原始解析度
    # val.py 在 640×640 算距離，predict CSV 是原始解析度
    # scale: original_w / 640（寬度方向）
    w, h = get_video_resolution(video_path)
    scale = w / 640
    tolerance = TOLERANCE_640 * scale

    vis_col = 'Visibility_orig' if 'Visibility_orig' in gt_df.columns else 'Visibility'
    gt_visible = gt_df[gt_df[vis_col] == 1].copy()

    # 建立 predict 的 frame → row 查詢
    pred_by_frame = {int(r['Frame']): r for _, r in pred_df.iterrows()}

    fn_rows = []
    for _, gt_row in gt_visible.iterrows():
        frame_idx = int(gt_row['Frame'])
        pred_row = pred_by_frame.get(frame_idx)

        is_fn = False
        fn_reason = ''

        if pred_row is None or float(pred_row['Conf']) < CONF_THRESHOLD:
            # conf < 0.5，完全沒偵測到 → FN
            is_fn = True
            fn_reason = 'no_detection'
        else:
            # conf >= 0.5，但檢查距離
            dist = np.sqrt((float(pred_row['X']) - float(gt_row['X']))**2 +
                           (float(pred_row['Y']) - float(gt_row['Y']))**2)
            if dist > tolerance:
                # 位置偏差 > tolerance → FP_dis（val 的定義）也算 FN
                is_fn = True
                fn_reason = 'wrong_position'

        if is_fn:
            row_dict = gt_row.to_dict()
            row_dict['fn_reason'] = fn_reason
            fn_rows.append(row_dict)

    if not fn_rows:
        return pd.DataFrame()

    fn_df = pd.DataFrame(fn_rows)
    fn_df['category'] = [classify_fn(row, gt_df, row['Frame']) for _, row in fn_df.iterrows()]
    return fn_df


# ===================== 抽樣連續幀 =====================

def draw_annotations(frame, gt_row, pred_row, frame_idx, is_fn_frame, trail_buffer=None):
    # 畫歷史軌跡（predict 軌跡，青→黃）
    if trail_buffer is not None and len(trail_buffer) > 0:
        n = len(trail_buffer)
        for ti, (tx, ty) in enumerate(trail_buffer):
            alpha = (ti + 1) / n
            radius = max(2, int(8 * alpha))
            brightness = int(255 * alpha)
            cv2.circle(frame, (tx, ty), radius, (0, brightness, brightness), -1)

    # GT（綠色）
    if gt_row is not None and int(gt_row.get('Visibility', 0)) == 1:
        gx, gy = int(gt_row['X']), int(gt_row['Y'])
        cv2.circle(frame, (gx, gy), 15, (0, 255, 0), 2)
        cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)
        cv2.putText(frame, 'GT', (gx + 18, gy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Predict（紅色）
    if pred_row is not None:
        px, py = int(pred_row['X']), int(pred_row['Y'])
        cv2.circle(frame, (px, py), 15, (0, 0, 255), 2)
        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
        cv2.putText(frame, f"P:{pred_row['Conf']:.2f}", (px + 18, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 左上角標註
    label = f"Frame:{frame_idx}" + ("  [FN]" if is_fn_frame else "")
    color = (0, 100, 255) if is_fn_frame else (200, 200, 200)
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # FN 幀加紅色邊框
    if is_fn_frame:
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 8)

    return frame


def sample_other_miss(all_fn_df, gt_dir, pred_dir, video_dir, output_dir,
                      n=50, context=10, seed=42, trail_length=0):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    video_dir = Path(video_dir)
    sample_dir = output_dir / 'sample_other_miss'
    sample_dir.mkdir(parents=True, exist_ok=True)

    other_miss = all_fn_df[all_fn_df['category'] == '其他漏偵測'].copy()
    print(f"\nOther Miss 總幀數：{len(other_miss)}")
    n = min(n, len(other_miss))
    sampled = other_miss.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"抽樣數量：{n}（每個前後各 {context} 幀）")

    saved_clips = 0
    for video_stem, group in sampled.groupby('video'):
        video_name_clean = video_stem.replace('_ball', '')
        video_path = video_dir / (video_name_clean + '.mp4')
        gt_path = gt_dir / (video_stem + '.csv')
        pred_path = pred_dir / video_name_clean / 'all.csv'

        if not video_path.exists():
            print(f"⚠ 找不到影片：{video_path}，跳過")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠ 無法開啟影片：{video_path}，跳過")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 讀 GT 和 predict，建立 frame → row 的快速查詢
        gt_by_frame = {}
        if gt_path.exists():
            gt_df = pd.read_csv(gt_path)
            gt_by_frame = {int(r['Frame']): r for _, r in gt_df.iterrows()}

        pred_by_frame = {}
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            pred_by_frame = {int(r['Frame']): r for _, r in pred_df.iterrows()}

        # 該影片所有 FN frame（不只被抽樣的），這樣 clip 裡其他 FN 也會標紅框
        video_all_fn = all_fn_df[all_fn_df['video'] == video_stem]
        fn_frames_set = set(int(f) for f in video_all_fn['Frame'].values)

        for _, row in group.iterrows():
            fn_frame = int(row['Frame'])
            clip_dir = sample_dir / f"{video_name_clean}_{fn_frame:06d}"
            clip_dir.mkdir(parents=True, exist_ok=True)

            start = max(0, fn_frame - context)
            end = min(total_frames - 1, fn_frame + context)

            trail_buffer = deque(maxlen=trail_length) if trail_length > 0 else None

            for fi in range(start, end + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue

                gt_row = gt_by_frame.get(fi)
                pred_row = pred_by_frame.get(fi)
                is_fn = fi in fn_frames_set

                # 更新軌跡 buffer
                if trail_buffer is not None and pred_row is not None and \
                        float(pred_row['Conf']) >= CONF_THRESHOLD:
                    trail_buffer.append((int(pred_row['X']), int(pred_row['Y'])))

                frame = draw_annotations(frame, gt_row, pred_row, fi, is_fn, trail_buffer)

                suffix = '_FN' if is_fn else ''
                cv2.imwrite(str(clip_dir / f"{fi:06d}{suffix}.jpg"), frame)

            saved_clips += 1

        cap.release()

    print(f"抽樣完成：{sample_dir}（共 {saved_clips} 個 clip）")


# ===================== 完整標註影片 =====================

def save_annotated_video(gt_path, pred_path, video_path, fn_frames_set, output_path,
                         trail_length=0):
    """輸出完整標註影片：GT（綠）、predict（紅）、FN 幀紅框、可選軌跡"""
    gt_df = pd.read_csv(gt_path)
    pred_df = pd.read_csv(pred_path) if Path(pred_path).exists() else pd.DataFrame()

    gt_by_frame = {int(r['Frame']): r for _, r in gt_df.iterrows()}
    pred_by_frame = {int(r['Frame']): r for _, r in pred_df.iterrows()} if len(pred_df) > 0 else {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠ 無法開啟影片：{video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    standard_fps = [30, 60, 90, 120, 150]
    fps = min(standard_fps, key=lambda x: abs(x - fps))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    trail_buffer = deque(maxlen=trail_length) if trail_length > 0 else None

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gt_row = gt_by_frame.get(fi)
        pred_row = pred_by_frame.get(fi)
        is_fn = fi in fn_frames_set

        # 更新軌跡 buffer
        if trail_buffer is not None and pred_row is not None and \
                float(pred_row['Conf']) >= CONF_THRESHOLD:
            trail_buffer.append((int(pred_row['X']), int(pred_row['Y'])))

        frame = draw_annotations(frame, gt_row, pred_row, fi, is_fn, trail_buffer)
        writer.write(frame)
        fi += 1

        if fi % 500 == 0:
            print(f"  {fi}/{total} frames...", end='\r')

    cap.release()
    writer.release()
    print(f"  標註影片已儲存：{output_path}")


# ===================== 主流程 =====================

def main(args):
    val_dir = Path(args.val_dir)
    gt_dir = val_dir / 'static_removal_before_csv'
    video_dir = val_dir / 'video'
    pred_dir = Path(args.pred_dir)

    # 自動加編號
    base_output_dir = Path(args.output_dir)
    idx = 1
    while True:
        output_dir = base_output_dir / f"{base_output_dir.name}{idx}"
        if not output_dir.exists():
            break
        idx += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    gt_files = sorted(gt_dir.glob('*.csv'))
    if len(gt_files) == 0:
        print(f"找不到 GT CSV 檔案：{gt_dir}")
        return

    all_fn, summary = [], []

    for gt_path in gt_files:
        video_name = gt_path.stem
        pred_path = pred_dir / video_name.replace('_ball', '') / 'all.csv'
        if not pred_path.exists():
            print(f"⚠ 找不到 predict CSV：{pred_path}，跳過")
            continue

        print(f"分析：{video_name}")
        video_name_clean = video_name.replace('_ball', '')
        video_path = video_dir / (video_name_clean + '.mp4')
        fn_df = analyze_video(gt_path, pred_path, video_path)

        if len(fn_df) == 0:
            print(f"  → 沒有 FN")
            continue

        fn_real = fn_df[fn_df['category'] != '正常靜止球']
        fn_normal_static = fn_df[fn_df['category'] == '正常靜止球']
        fn_df['video'] = video_name
        all_fn.append(fn_df)

        counts = fn_real['category'].value_counts()
        gt_df_reload = pd.read_csv(gt_path)
        total_gt = len(gt_df_reload[gt_df_reload['Visibility'] == 1])

        print(f"  GT 有球幀數：{total_gt}")
        print(f"  FN（不含正常靜止球）：{len(fn_real)}")
        print(f"  正常靜止球（不計入FN）：{len(fn_normal_static)}")
        for cat, cnt in counts.items():
            print(f"    {cat}: {cnt}")

        summary.append({
            'video': video_name,
            'gt_visible': total_gt,
            'fn_total': len(fn_real),
            **{cat: counts.get(cat, 0) for cat in ['接球瞬間', '高速運動模糊', 'rally中靜止球', '其他漏偵測']}
        })

        # 輸出完整標註影片
        if args.save_annotated_video:
            fn_frames_set = set(int(f) for f in fn_real['Frame'].values)
            out_path = output_dir / 'annotated_video' / (video_name_clean + '_annotated.mp4')
            print(f"  輸出標註影片：{out_path.name}")
            save_annotated_video(gt_path, pred_path, video_path, fn_frames_set, out_path,
                                 trail_length=args.trail_length)

    if not all_fn:
        print("沒有任何 FN 資料")
        return

    all_fn_df = pd.concat(all_fn, ignore_index=True)
    all_fn_df.to_csv(output_dir / 'fn_all.csv', index=False)
    pd.DataFrame(summary).to_csv(output_dir / 'fn_summary.csv', index=False)
    print(f"\n所有 FN 已儲存：{output_dir / 'fn_all.csv'}")
    print(f"Summary 已儲存：{output_dir / 'fn_summary.csv'}")

    real_fn = all_fn_df[all_fn_df['category'] != '正常靜止球']
    if len(real_fn) > 0:
        label_map = {
            '其他漏偵測': 'Other Miss',
            '高速運動模糊': 'Motion Blur',
            'rally中靜止球': 'Static in Rally',
            '接球瞬間': 'Hit Moment',
        }
        counts = real_fn['category'].value_counts()
        counts.index = [label_map.get(c, c) for c in counts.index]

        plt.figure(figsize=(8, 5))
        counts.plot(kind='bar', color=['#e74c3c', '#e67e22', '#3498db', '#95a5a6'])
        plt.title('FN Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Frame Count')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'fn_category_chart.png', dpi=150)
        plt.close()
        print(f"統計圖已儲存：{output_dir / 'fn_category_chart.png'}")
        print("\n===== 總體 FN 統計 =====")
        print(counts.to_string())
        print(f"總 FN 幀數：{len(real_fn)}")

        sample_other_miss(all_fn_df, gt_dir, pred_dir, video_dir, output_dir,
                          n=args.sample_n, context=args.context, seed=args.seed,
                          trail_length=args.trail_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str, required=True,
                        help='val_data 下的資料集資料夾（e.g. sportxai_rally_test）')
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='/usr/src/ultralytics/runs/analysis')
    parser.add_argument('--save_annotated_video', action='store_true',
                        help='輸出完整標註影片（GT綠色、predict紅色、FN幀紅框）')
    parser.add_argument('--trail_length', type=int, default=0,
                        help='軌跡顯示幀數（0=關閉，建議值=30），影片和抽樣圖片都適用')
    parser.add_argument('--sample_n', type=int, default=50,
                        help='Other Miss 抽樣數量（預設 50）')
    parser.add_argument('--context', type=int, default=10,
                        help='FN 前後各幾幀（預設 10）')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)