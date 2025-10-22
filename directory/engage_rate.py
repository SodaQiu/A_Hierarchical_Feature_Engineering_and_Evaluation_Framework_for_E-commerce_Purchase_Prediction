import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

# --- 1. 加载数据 ---
file_path = r"C:\Users\47556\Desktop\no_day1.xlsx"
df = pd.read_excel(file_path)

TARGET = 'buy_yn'

# --- 2. 构造衍生特征 ---
df['cart_rate'] = df['cart_avg'] / (df['pv_avg'] + 1e-6)
df['fav_rate'] = df['fav_avg'] / (df['pv_avg'] + 1e-6)
df['engage_rate'] = (df['cart_avg'] + df['fav_avg']) / (df['pv_avg'] + 1e-6)

# --- 3. 准备特征和目标 ---
drop_cols = [TARGET, 'user_id', 'date', 'buy_min', 'buy_max', 'buy_avg']  # 删除目标和无关列
X = df.drop(columns=drop_cols, errors="ignore")
y = df[TARGET]

# 标准化（部分模型会更稳定）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. 定义模型 ---
# Base 模型 (RF + XGB)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, eval_metric="logloss")

# Mate (Stacking: LR + RF + XGB)
stack_model = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb)],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    stack_method="predict_proba",
    passthrough=False,
    cv=5
)

models = {
    "Base: RF": rf,
    "Base: XGB": xgb,
    "Mate: LR+Stacking(RF+XGB)": stack_model
}

# --- 5. 5折交叉验证评估 ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y):
    accs, f1s, aucs, recalls = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_proba))
        recalls.append(recall_score(y_val, y_pred))

    return {
        "ACC": (np.mean(accs), np.std(accs)),
        "F1": (np.mean(f1s), np.std(f1s)),
        "AUC": (np.mean(aucs), np.std(aucs)),
        "Recall": (np.mean(recalls), np.std(recalls))
    }

# --- 6. 输出结果 ---
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_scaled, y)

# 打印结果表格
print("\n结果对比：")
print("{:<30} {:<20} {:<20} {:<20} {:<20}".format("模型", "Accuracy", "F1", "AUC", "Recall"))
for name, res in results.items():
    print("{:<30} {:.4f} (+/-{:.4f})   {:.4f} (+/-{:.4f})   {:.4f} (+/-{:.4f})   {:.4f} (+/-{:.4f})".format(
        name,
        res["ACC"][0], res["ACC"][1],
        res["F1"][0], res["F1"][1],
        res["AUC"][0], res["AUC"][1],
        res["Recall"][0], res["Recall"][1]
    ))
