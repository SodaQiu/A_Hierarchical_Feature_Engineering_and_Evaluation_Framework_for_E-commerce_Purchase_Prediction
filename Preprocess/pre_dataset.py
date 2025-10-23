import pandas as pd

# 输入/输出路径
in_path  = r"C:\Users\47556\Desktop\user_time_with_user_stats.xlsx"
out_path = r"C:\Users\47556\Desktop\no_day1.xlsx"

# 读取
df = pd.read_excel(in_path)
print("原始列：", list(df.columns))

# 1) 删除无关列
drop_cols = [c for c in ['suspected_spider', 'day_type', 'time_period'] if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print("已删除：", drop_cols)

# 2) 统一列名：把 *_count 改成最终需要的列名
rename_map = {}
if 'cart_count' in df.columns: rename_map['cart_count'] = 'cart'
if 'fav_count'  in df.columns: rename_map['fav_count']  = 'fav'
if 'buy_count'  in df.columns: rename_map['buy_count']  = 'buy'
# 注意：pv_count 按你的截图保留这个名字，不改
df = df.rename(columns=rename_map)

# 3) 按用户聚合（若每个 user_id 多行）
agg_dict = {}

# —— 计数列：求和 ——（存在则聚合）
for c in ['pv_count', 'cart', 'fav', 'buy']:
    if c in df.columns:
        agg_dict[c] = 'sum'

# —— 分类标签：buy_yn（若存在），聚合为：只要有一次购买就置1 ——
if 'buy_yn' in df.columns:
    agg_dict['buy_yn'] = 'max'

# —— 统计列：*_min 取最小、*_max 取最大、*_avg 取平均 ——
for prefix in ['pv', 'cart', 'fav', 'buy']:
    c_min, c_max, c_avg = f'{prefix}_min', f'{prefix}_max', f'{prefix}_avg'
    if c_min in df.columns: agg_dict[c_min] = 'min'
    if c_max in df.columns: agg_dict[c_max] = 'max'
    if c_avg in df.columns: agg_dict[c_avg] = 'mean'

# 如果没有需要聚合的列，也至少保留 user_id
if not agg_dict:
    agg_dict = {col: 'first' for col in df.columns if col != 'user_id'}

# 实施聚合
if 'user_id' not in df.columns:
    raise ValueError("缺少 user_id 列，无法合并")

df = df.groupby('user_id', as_index=False).agg(agg_dict)

# 4) buy_yn 补全
if 'buy_yn' not in df.columns and 'buy' in df.columns:
    df['buy_yn'] = (df['buy'] > 0).astype(int)

# 5) 调整列顺序
cols_order = [
    'user_id',
    'cart', 'fav', 'buy',
    'buy_yn',
    'pv_min', 'pv_max', 'pv_avg',
    'cart_min', 'cart_max', 'cart_avg',
    'fav_min', 'fav_max', 'fav_avg',
    'buy_min', 'buy_max', 'buy_avg',
    'pv_count'
]
df = df[[c for c in cols_order if c in df.columns]]

# 6) 保存
df.to_excel(out_path, index=False)
print(f"处理完成，已保存到：{out_path}")
print("最终列：", list(df.columns))

# 7)（可选）检查 buy_target 兼容性
if 'buy' in df.columns:
    df['buy_target'] = (df['buy'] > 0).astype(int)
    print("success")
else:
    print("error")



