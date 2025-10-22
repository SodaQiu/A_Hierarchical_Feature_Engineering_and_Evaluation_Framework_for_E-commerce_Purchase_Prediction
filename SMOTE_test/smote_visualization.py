# --- 第 1 步: 导入所有需要的库 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# --- 第 2 步: 定义一个主函数来整合所有逻辑 ---
def create_smote_comparison_plot(file_path):
    """
    加载数据，应用SMOTE，并使用PCA可视化前后的数据分布。
    """
    # --- 数据加载与预处理 ---
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径 '{file_path}' 是否正确。")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    df = df.fillna(0)

    if 'buy_yn' in df.columns:
        df['buy_target'] = (df['buy_yn'] > 0).astype(int)
    elif 'buy' in df.columns:
        df['buy_target'] = (df['buy'] > 0).astype(int)
    else:
        raise ValueError("错误：数据中必须包含 'buy_yn' 或 'buy' 列来定义目标变量。")

    print("\n===== 数据检查 =====")
    print("构造的 'buy_target' 分布：")
    print(df['buy_target'].value_counts())

    # --- 定义特征和目标变量 (动态识别方式) ---
    print("\n===== 动态识别特征列 =====")
    # 1. 定义需要从特征中排除的列
    exclude_cols = ['buy_target', 'buy_yn', 'buy', 'user_id']

    # 2. 动态获取所有特征列的名称
    feature_names = [col for col in df.columns if col not in exclude_cols]

    print(f"成功识别出 {len(feature_names)} 个特征。")

    # 从DataFrame中分离出特征 X 和目标 y
    X = df[feature_names]
    y = df['buy_target']

    print("\n原始数据集（处理前）的类别分布:")
    print(sorted(Counter(y).items()))

    # --- 数据标准化 ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 应用 SMOTE 来平衡数据 ---
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print("\n经过 SMOTE 处理后的数据集类别分布:")
    print(sorted(Counter(y_resampled).items()))

    # --- 使用 PCA 进行降维 ---
    pca = PCA(n_components=2)
    X_pca_before = pca.fit_transform(X_scaled)
    X_pca_after = pca.fit_transform(X_resampled)

    # --- 可视化对比图 ---
    colors = {0: 'blue', 1: 'red'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: Before SMOTE
    counts_before = Counter(y)
    title_before = f"Before SMOTE\nClass 0 (No-buy): {counts_before[0]} | Class 1 (Buy): {counts_before[1]}"
    for label in np.unique(y):
        row_ix = np.where(y == label)[0]
        ax1.scatter(
            X_pca_before[row_ix, 0], X_pca_before[row_ix, 1],
            label=f'Class {label}', color=colors[label], alpha=0.7
        )
    ax1.set_title(title_before)
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.legend()

    # 右图: After SMOTE
    counts_after = Counter(y_resampled)
    title_after = f"After SMOTE\nClass 0 (No-buy): {counts_after[0]} | Class 1 (Buy): {counts_after[1]}"
    scatter_after = ax2.scatter(
        X_pca_after[:, 0], X_pca_after[:, 1],
        c=y_resampled, cmap='coolwarm', alpha=0.7
    )
    ax2.set_title(title_after)
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    handles, _ = scatter_after.legend_elements()
    ax2.legend(handles=handles, labels=['Class 0', 'Class 1'], title="Class")

    fig.suptitle("Data Distribution Before and After SMOTE (PCA Reduced)", fontsize=16, y=1.02)
    plt.tight_layout()

    # --- 保存或显示图片 ---
    output_filename = 'smote_pca_comparison.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\n对比图已保存为 '{output_filename}'")

    # plt.show()

# --- 第 3 步: 设置文件路径并执行主函数 ---
if __name__ == "__main__":
    # ##################################################
    # ### 请将这里的文件路径修改为您自己的 Excel 文件路径 ###
    # ##################################################
    file_path = r'C:\Users\47556\Desktop\no_day1.xlsx'

    if not os.path.exists(file_path):
        print(f"致命错误：文件 '{file_path}' 不存在。请确认路径是否正确。")
    else:
        create_smote_comparison_plot(file_path)