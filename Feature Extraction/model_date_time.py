import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

# --- 1. 加载数据 ---
# 请将 'your_data.csv' 替换为您的数据文件路径
file_path = r"C:\Users\47556\Desktop\user_time_with_user_stats.xlsx"
df = pd.read_excel(file_path)

# --- 2. 特征工程 ---

# A. 处理 'day_type' 列: 将文本映射为数字
# 'weekday' -> 0, 'weekend' -> 1
day_type_mapping = {'Weekday': 0, 'Weekend': 1}
df['day_type'] = df['day_type'].map(day_type_mapping)


# B. 处理 'time_period' 列: 按照您提供的字典进行精确映射
time_mapping = {
    'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Late Night': 4,
    '1': 1, '2': 2, '3': 3, '4': 4,
    1: 1, 2: 2, 3: 3, 4: 4
}
df['time_period'] = df['time_period'].map(time_mapping)

print("数据处理完成，以下是处理后数据的前几行，请检查：")
print(df.head())
# --- 3. 数据准备 ---

# 定义目标变量
TARGET = 'buy_yn'

# 定义需要移除的特征（根据您的新表头）
# user_id 是标识符，buy_count 用于防止数据泄露
features_to_drop = ['user_id', 'buy_count']
df_cleaned = df.drop(columns=features_to_drop, errors='ignore')

# 检查空值
if df_cleaned.isnull().sum().any():
    print("\n警告：数据中存在空值，将使用众数填充。")
    df_cleaned = df_cleaned.fillna(df_cleaned.mode().iloc[0])

# 分离特征 (X) 和目标 (y)
X = df_cleaned.drop(columns=[TARGET], errors='ignore')
y = df_cleaned[TARGET]

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- 4. 初始化模型 ---
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

# --- 5. 5折交叉验证与评估 ---
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n--- 正在评估: {name} ---")

    acc_scores, f1_scores, auc_scores, recall_scores = [], [], [], []

    X_to_use = X_scaled if name == "Logistic Regression" else X.values

    for train_idx, val_idx in cv.split(X_to_use, y):
        X_train, X_val = X_to_use[train_idx], X_to_use[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        recall_scores.append(recall_score(y_val, y_pred))

    print(f"  准确率 (Accuracy): {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"  F1分数 (F1-Score):  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    print(f"  AUC:              {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"  召回率 (Recall):    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
    print("="*45)