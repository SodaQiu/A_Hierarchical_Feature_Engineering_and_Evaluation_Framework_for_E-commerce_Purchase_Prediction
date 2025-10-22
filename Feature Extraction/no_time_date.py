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

file_path = r"C:\Users\47556\Desktop\user_time_with_user_stats.xlsx"
df = pd.read_excel(file_path)

day_type_mapping = {'Weekday': 0, 'Weekend': 1}
df['day_type'] = df['day_type'].map(day_type_mapping)

time_mapping = {
    'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Late Night': 4,
    '1': 1, '2': 2, '3': 3, '4': 4,
    1: 1, 2: 2, 3: 3, 4: 4
}
df['time_period'] = df['time_period'].map(time_mapping)
# --- 3. 数据准备 ---

# 定义目标变量
TARGET = 'buy_yn'
features_to_drop = ['user_id','buy_count', 'buy', 'day_type', 'time_period', 'buy_min', 'buy_max', 'buy_avg']
df_cleaned = df.drop(columns=features_to_drop, errors='ignore')

if df_cleaned.isnull().sum().any():
    df_cleaned = df_cleaned.fillna(df_cleaned.mode().iloc[0])

X = df_cleaned.drop(columns=[TARGET], errors='ignore')
y = df_cleaned[TARGET]

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

# --- 5. 执行5折交叉验证并输出平均结果 ---

# 定义5折交叉验证器
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 对每个模型进行评估
for name, model in models.items():
    print(f"\n--- 正在评估 去掉date time: {name} ---")

    # 创建列表，用于存储每一折的评估分数
    acc_scores, f1_scores, auc_scores, recall_scores = [], [], [], []

    X_to_use = X_scaled if name == "Logistic Regression" else X.values

    # cv.split 会生成5组训练集和验证集的索引，循环5次
    for train_idx, val_idx in cv.split(X_to_use, y):
        # 第一折：train_idx是后80%数据，val_idx是前20%数据
        # 第二折：train_idx是另外80%数据，val_idx是另外20%数据...以此类推

        # 根据索引划分出当前这一折的训练集和验证集
        X_train, X_val = X_to_use[train_idx], X_to_use[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 在当前训练集上训练模型
        model.fit(X_train, y_train)

        # 在当前验证集上进行预测和评估
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # 将这一折的结果存入列表
        acc_scores.append(accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        recall_scores.append(recall_score(y_val, y_pred))

    # 循环结束后，列表里有5个分数，计算它们的平均值和标准差
    print(f"  准确率 (Accuracy): {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"  F1分数 (F1-Score):  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    print(f"  AUC:              {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"  召回率 (Recall):    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
    print("="*45)