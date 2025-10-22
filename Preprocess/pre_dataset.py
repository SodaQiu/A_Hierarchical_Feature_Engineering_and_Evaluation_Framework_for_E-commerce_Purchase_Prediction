import pandas as pd

# 读取 Excel 文件
file_path = r"C:\Users\47556\Desktop\user_time_with_user_stats.xlsx"
df = pd.read_excel(file_path)

print(f"原始数据形状: {df.shape}")
print(f"原始列名: {list(df.columns)}")

# 1. 删除 day 列（如果存在）
if 'day_type' in df.columns:
    df = df.drop(columns=['day_type'])
    print("已删除 day_type 列")
else:
    print("未找到 day_type 列")

# 2. 处理时间列：将时间转换为整型
if 'time_period' in df.columns:
    print(f"\ntime_period列数据类型: {df['time_period'].dtype}")
    print(f"time_period列唯一值: {df['time_period'].unique()}")

    # 时间映射规则
    time_mapping = {

        'Morning': 1, 'Afternoon': 2, 'Evening': 3,'Late Night': 4,
        '1': 1, '2': 2, '3': 3,'4': 4,
        1: 1, 2: 2, 3: 3,4: 4
    }

    if df['time_period'].dtype == 'object':
        # 清理数据并转换
        df['time_period'] = df['time_period'].astype(str).str.strip()
        df['time_period'] = df['time_period'].map(time_mapping)

        # 检查未映射的值
        unmapped_mask = df['time_period'].isna()
        if unmapped_mask.any():
            print(f"警告：发现 {unmapped_mask.sum()} 个无法映射的时间值")
            df['time_period'] = df['time_period'].fillna(2)  # 默认为中午

    # 确保time_period列为整型
    df['time_period'] = df['time_period'].astype(int)
    print(f"时间列已转换为整型，唯一值: {sorted(df['time_period'].unique())}")
else:
    print("未找到 time_period 列")

# 3. 按 user_id 合并不同 time_period 的数据
if 'user_id' in df.columns:
    print(f"\n开始按 user_id 合并数据...")

    # 定义聚合函数
    agg_functions = {}

    # 指定需要取第一行数据的列（统计特征列）
    first_cols = ['pv_min', 'pv_max', 'pv_avg', 'cart_min', 'cart_max', 'cart_avg',
                  'fav_min', 'fav_max', 'fav_avg', 'buy_min', 'buy_max', 'buy_avg']

    # 指定需要求和的列（原始行为数据）
    sum_cols = ['pv', 'cart', 'fav', 'buy']

    # 找出所有pv相关列（用于求和）
    pv_cols = [col for col in df.columns if 'pv' in col.lower() and col not in first_cols]

    for col in df.columns:
        if col != 'user_id':
            if col in first_cols:
                agg_functions[col] = 'first'  # 统计特征列取第一个值
            elif col in sum_cols:
                agg_functions[col] = 'sum'  # 原始行为数据求和
            elif 'pv' in col.lower():
                agg_functions[col] = 'sum'  # 其他pv相关列求和
            else:
                agg_functions[col] = 'first'  # 其他列取第一个值

    # 按 user_id 聚合
    user_aggregated = df.groupby('user_id').agg(agg_functions).reset_index()

    # 将需要求和的pv相关列合并为一个pv_count列
    if pv_cols:
        user_aggregated['pv_count'] = user_aggregated[pv_cols].sum(axis=1)
        # 删除原来的pv列
        user_aggregated = user_aggregated.drop(columns=pv_cols)

    # 删除time_period列，因为已经按user_id合并了不同时间段的数据
    if 'time_period' in user_aggregated.columns:
        user_aggregated = user_aggregated.drop(columns=['time_period'])
        print("已删除time_period列（数据已按user_id合并）")

    print(f"聚合前数据行数: {len(df)}")
    print(f"聚合后数据行数: {len(user_aggregated)}")
    print(f"求和的pv相关列: {pv_cols}")
    print(f"求和的原始行为列: {[col for col in sum_cols if col in df.columns]}")
    print(f"取第一行数据的统计特征列: {[col for col in first_cols if col in df.columns]}")

    # 使用聚合后的数据
    df = user_aggregated

    print(f"\n合并后数据形状: {df.shape}")
    print(f"合并后列名: {list(df.columns)}")

else:
    print(f"警告: 数据中缺少 user_id 列，无法进行合并")
    print(f"数据列名: {list(df.columns)}")

# 生成新的 Excel 文件
# output_path = r"E:\google\Lab\alibaba\no_day2.xlsx"
output_path = r'C:\Users\47556\Desktop\no_day1.xlsx'
df.to_excel(output_path, index=False)

print(f"\n处理完成，结果已保存到: {output_path}")
print(f"最终数据形状: {df.shape}")
print(f"最终列名: {list(df.columns)}")
if 'user_id' in df.columns and 'time_period' in df.columns:
    print(f"用户数量: {df['user_id'].nunique()}")
    print(f"时间段分布: {df['time_period'].value_counts().sort_index().to_dict()}")
    print(f"用户-时间组合数: {len(df)}")

