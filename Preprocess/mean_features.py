import pandas as pd
import os

# 1) 读取清洗后的明细数据
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
input_file = os.path.join(desktop, "cleaned_dataset.xlsx")
df = pd.read_excel(input_file)

# 你的列应该是：user_id, day_type, time_period, pv_count, cart_count, fav_count, buy_count
needed = ["user_id", "day_type", "time_period", "pv_count", "cart_count", "fav_count", "buy_count"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"缺少必要列: {missing}")

# 确保数值型
for c in ["pv_count", "cart_count", "fav_count", "buy_count"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# 2) 按 user_id 计算用户级统计
agg_dict = {
    "pv_count": ["min", "max", "mean"],
    "cart_count": ["min", "max", "mean"],
    "fav_count": ["min", "max", "mean"],
    "buy_count": ["min", "max", "mean"],
}
user_stats = df.groupby("user_id").agg(agg_dict)

# 3) 展平列名
rename_map = {
    ("pv_count", "min"): "pv_min",
    ("pv_count", "max"): "pv_max",
    ("pv_count", "mean"): "pv_avg",
    ("cart_count", "min"): "cart_min",
    ("cart_count", "max"): "cart_max",
    ("cart_count", "mean"): "cart_avg",
    ("fav_count", "min"): "fav_min",
    ("fav_count", "max"): "fav_max",
    ("fav_count", "mean"): "fav_avg",
    ("buy_count", "min"): "buy_min",
    ("buy_count", "max"): "buy_max",
    ("buy_count", "mean"): "buy_avg",
}
user_stats.columns = [rename_map[col] for col in user_stats.columns]
user_stats = user_stats.reset_index()

# 可选：均值保留两位小数
for col in [c for c in user_stats.columns if c.endswith("_avg")]:
    user_stats[col] = user_stats[col].round(2)

# 4) 合并回原始表
df_out = df.merge(user_stats, on="user_id", how="left")

# 5) 保存
output_file = os.path.join(desktop, "user_time_with_user_stats.xlsx")
df_out.to_excel(output_file, index=False)

print(f"结果已保存到: {output_file}")
print(df_out.head())
