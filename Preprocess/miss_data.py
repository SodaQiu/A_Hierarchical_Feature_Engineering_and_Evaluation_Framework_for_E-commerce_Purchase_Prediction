import pandas as pd
import os

df = pd.read_excel(r"C:\Users\47556\Desktop\dataset_for_prediction.xlsx")

# 1. 聚合用户行为
pv_threshold = df["pv_count"].quantile(0.99)    # 浏览99分位
buy_threshold = df["buy_count"].quantile(0.99)  # 购买99分位

# 2. 定义规则
cond_high_pv_zero_buy = (df["pv_count"] > pv_threshold) & (df["buy_count"] == 0)
cond_high_buy_low_pv  = (df["buy_count"] > buy_threshold) & (df["pv_count"] < 5)

# 3. 标记可疑用户
df["suspected_spider"] = 0
df.loc[cond_high_pv_zero_buy | cond_high_buy_low_pv, "suspected_spider"] = 1


df_clean = df[df["suspected_spider"] == 0].copy()

# 4. 统计结果
num_total = len(df)
num_spiders = df["suspected_spider"].sum()
print(f"疑似爬虫用户数量: {num_spiders}")
ratio_spiders = num_spiders / num_total * 100

print(f"总用户数: {num_total}")
print(f"疑似爬虫用户数: {num_spiders}")
print(f"疑似爬虫用户比例: {ratio_spiders:.4f}%")

# 可选：分别看两类异常的数量和比例
num_pv_spiders = cond_high_pv_zero_buy.sum()
num_buy_spiders = cond_high_buy_low_pv.sum()

print("\n--- 异常类型细分 ---")
print(f"高浏览零购买用户数: {num_pv_spiders} ({num_pv_spiders/num_total*100:.4f}%)")
print(f"高购买低浏览用户数: {num_buy_spiders} ({num_buy_spiders/num_total*100:.4f}%)")

# 输出前几条可疑用户数据
print(df[df["suspected_spider"] == 1].head())

desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# 保存 Excel
output_excel = os.path.join(desktop, "cleaned_dataset.xlsx")
df_clean.to_excel(output_excel, index=False)

# 保存 CSV (可选)
output_csv = os.path.join(desktop, "cleaned_dataset.csv")
df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")


print(f"\n清洗后的数据集已保存到：\n{output_excel}\n{output_csv}")
