import pandas as pd
df = pd.read_csv('/usr/src/ultralytics/runs/detect/predict76/predict_data/csv/videoplayback1/all.csv')

gaps = df['Frame'].diff()
large_gaps = gaps[gaps > 10]
print(large_gaps.to_string())  # 不截斷
print(f"\n總共 {len(large_gaps)} 個斷點")
print(f"最大斷點: {large_gaps.max():.0f} 幀")
print(f"平均斷點: {large_gaps.mean():.1f} 幀")

print(large_gaps.value_counts().sort_index())
