#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_add_empty_frames_with_y_adjust_from0.py

同前版本，但輸出 Frame 範圍從 0 開始直到原資料的 max(Frame)。

Usage:
    python convert_add_empty_frames_with_y_adjust_from0.py input.csv
    python convert_add_empty_frames_with_y_adjust_from0.py input.csv --out out.csv --fps 30
"""
import argparse
import os
import sys
from datetime import datetime
import pandas as pd

OUT_HEADER = ["Frame", "Visibility", "X", "Y", "Z", "Event", "Timestamp"]

def parse_args():
    p = argparse.ArgumentParser(description="Convert CSV and fill missing frames; apply Y_adjust = Y - 80.0. Output frames start from 0.")
    p.add_argument("input", help="Input CSV path (must contain Frame, Visibility, X, Y columns).")
    p.add_argument("--out", "-o", help="Output CSV file path. If omitted, will write to current dir with timestamped name.")
    p.add_argument("--outdir", help="If --out omitted, use this directory for generated filename. Default '.'", default=".")
    p.add_argument("--fps", type=float, default=30.0, help="Frame rate to compute Timestamp = Frame / fps (default 30.0).")
    p.add_argument("--ts-mult", type=float, default=1.0, help="Multiply timestamp by this factor (default 1.0).")
    p.add_argument("--no-fill", action="store_true", help="Do NOT fill missing frames (only convert existing frames).")
    return p.parse_args()

def main():
    args = parse_args()
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(2)

    # required columns
    for c in ("Frame", "Visibility", "X", "Y"):
        if c not in df.columns:
            print(f"Error: input CSV missing required column: {c}", file=sys.stderr)
            sys.exit(2)

    # coerce types
    df = df.copy()
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df = df[df["Frame"].notna()]  # drop rows with non-numeric frame
    df["Frame"] = df["Frame"].astype(int)
    df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce").fillna(0).astype(int)
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

    # Group by Frame:
    grouped = {}
    for frame, g in df.groupby("Frame"):
        g_vis1 = g[g["Visibility"] == 1]
        if len(g_vis1) > 0:
            mean_x = float(g_vis1["X"].mean())
            mean_y = float(g_vis1["Y"].mean())
            visibility = 1
        else:
            mean_x = float(g["X"].mean()) if not g["X"].isna().all() else float("nan")
            mean_y = float(g["Y"].mean()) if not g["Y"].isna().all() else float("nan")
            visibility = 0
        grouped[int(frame)] = {"Visibility": visibility, "X": mean_x, "Y": mean_y}

    frames_present = sorted(grouped.keys())
    if len(frames_present) == 0:
        print("No frames found in input.", file=sys.stderr)
        sys.exit(2)

    # IMPORTANT CHANGE: start from frame 0
    min_frame = 0
    max_frame = frames_present[-1]

    rows = []
    if args.no_fill:
        frame_iter = frames_present
    else:
        frame_iter = range(min_frame, max_frame + 1)

    Y_ADJUST = (640.0 - 480.0) / 2.0  # = 80.0

    for f in frame_iter:
        if f in grouped:
            vis = grouped[f]["Visibility"]
            x = grouped[f]["X"]
            y = grouped[f]["Y"]
            # if grouped X/Y are NaN and vis==0, set to 0.0 to have numeric CSV
            if (not pd.notna(x) or not pd.notna(y)) and vis == 0:
                x_out = 0.0
                y_out = 0.0
            else:
                x_val = float(x) if pd.notna(x) else 0.0
                y_val = float(y) if pd.notna(y) else 0.0
                # apply Y adjustment only when there is a valid coordinate (i.e., vis==1)
                if vis == 1:
                    y_out = y_val - Y_ADJUST
                else:
                    # keep 0.0 for missing / invisible frames
                    y_out = y_val if (pd.notna(y) and vis == 0 and y_val != 0.0) else 0.0
                x_out = x_val
        else:
            # missing frame -> Visibility 0, zero coords
            vis = 0
            x_out = 0.0
            y_out = 0.0

        z_out = 0.0
        event_out = 0
        timestamp = (float(f) / float(args.fps)) * float(args.ts_mult)
        rows.append({
            "Frame": int(f),
            "Visibility": int(vis),
            "X": float(x_out),
            "Y": float(y_out),
            "Z": float(z_out),
            "Event": int(event_out),
            "Timestamp": float(timestamp)
        })

    out_df = pd.DataFrame(rows, columns=OUT_HEADER)

    # prepare output path
    if args.out:
        out_path = args.out
    else:
        os.makedirs(args.outdir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(args.outdir, f"{base}_converted_yadjust_from0_{now}.csv")

    try:
        out_df.to_csv(out_path, index=False, float_format="%.6f")
    except Exception as e:
        print(f"Error writing output CSV: {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Wrote {len(out_df)} rows to: {out_path}")

if __name__ == "__main__":
    main()
