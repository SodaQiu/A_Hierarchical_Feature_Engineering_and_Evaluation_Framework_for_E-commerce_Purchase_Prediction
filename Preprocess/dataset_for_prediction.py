import pandas as pd
import os
from datetime import datetime

# ========= 配置：修改为你的原始数据路径（csv 或 xlsx 都可） =========
input_path = r"C:\Users\47556\Desktop\UserBehavior.csv"  # 或 .csv
# 输出到同目录
output_path = os.path.join(os.path.dirname(input_path), "dataset_for_prediction.xlsx")

# ========= 1) 读取并添加列名 =========
if input_path.lower().endswith(".csv"):
    df = pd.read_csv(input_path, header=None)
else:
    df = pd.read_excel(input_path, header=None)

df.columns = ["user_id", "item_id", "cate_id", "behavior", "timestamp"]

# ========= 2) 解析时间戳（北京时间，精确到小时）=========
def parse_to_beijing(ts):
    """把数字时间戳解析为北京时间整小时"""
    s = str(ts).strip()
    if s.isdigit():
        v = int(s)
        # 判断毫秒/秒
        if v > 10**12:
            dt = pd.to_datetime(v, unit="ms", utc=True)
        elif v > 10**10:
            dt = pd.to_datetime(v, unit="ms", utc=True)
        else:
            dt = pd.to_datetime(v, unit="s", utc=True)
    else:
        dt = pd.to_datetime(s, errors="coerce", utc=True)

    if pd.isna(dt):
        return pd.NaT

    # 转北京时间（Asia/Shanghai）并取整到小时
    return dt.tz_convert("Asia/Shanghai").tz_localize(None).floor("h")

df["timestamp"] = df["timestamp"].apply(parse_to_beijing)
if df["timestamp"].isna().any():
    bad = df["timestamp"].isna().sum()
    raise ValueError(f"有 {bad} 条无法解析的时间戳，请检查原始数据。")


start_date = pd.Timestamp("2017-11-25 00:00:00")
end_date   = pd.Timestamp("2017-12-03 23:59:59")

df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
print(f"筛选后数据行数: {len(df)}")

# ========= 3) day_type（工作日/周末） & time_period（时段）=========
df["day_type"] = df["timestamp"].dt.weekday.map(lambda d: "Weekend" if d >= 5 else "Weekday")

def to_period(h):
    if 0 <= h <= 5:
        return "Early Morning"
    elif 6 <= h <= 11:
        return "Morning"
    elif 12 <= h <= 17:
        return "Afternoon"
    else:
        return "Late Night"

df["time_period"] = df["timestamp"].dt.hour.map(to_period)

# ========= 4) 按 user_id + day_type + time_period 聚合行为次数 =========
agg = (
    df.groupby(["user_id", "day_type", "time_period", "behavior"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# 确保四种行为都存在
for b in ["pv", "cart", "fav", "buy"]:
    if b not in agg.columns:
        agg[b] = 0

# 重命名 *_count
agg = agg.rename(columns={
    "pv": "pv_count",
    "cart": "cart_count",
    "fav": "fav_count",
    "buy": "buy_count"
})

# 添加 buy_yn（二分类标签）
agg["buy_yn"] = (agg["buy_count"] > 0).astype(int)

# ========= 5) 调整列顺序并保存 =========
final_cols = [
    "user_id", "day_type", "time_period",
    "pv_count", "cart_count", "fav_count", "buy_count",
    "buy_yn"
]
agg = agg[final_cols].sort_values(["user_id", "day_type", "time_period"], ignore_index=True)

agg.to_excel(output_path, index=False)
print(f"✅ 已生成：{output_path}")
print(agg.head())
