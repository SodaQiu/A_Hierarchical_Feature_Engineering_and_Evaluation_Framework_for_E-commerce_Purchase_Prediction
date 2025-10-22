import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE

# =============================
# 1) 读取数据
# =============================
file_path = r"C:\Users\47556\Desktop\no_day1.xlsx"
df = pd.read_excel(file_path)

TARGET = 'buy_yn'
if TARGET not in df.columns:
    raise ValueError(f"未找到目标列 {TARGET}")

# =============================
# 2) 构造/补齐 median 特征
#   变量：pv, fav, cart_count
#   已有：*_min, *_max, *_avg
#   需要：*_median（若缺，就用 *_avg 近似）
# =============================
def ensure_median(col_base: str):
    """若不存在 {col_base}_median，则用 {col_base}_avg 近似生成，并提示。"""
    med_col = f"{col_base}_median"
    avg_col = f"{col_base}_avg"
    if med_col not in df.columns:
        if avg_col in df.columns:
            df[med_col] = df[avg_col]
            print(f"[提示] 未检测到列 {med_col}，已使用 {avg_col} 作为近似值生成。")
        else:
            # 如果连 *_avg 都没有，就报错提示用户检查字段
            raise ValueError(f"未找到 {med_col} 和 {avg_col}，请检查数据列是否完整。")

for base in ["pv", "fav", "cart"]:
    ensure_median(base)

# =============================
# 3) 选择特征列 & 清理
# =============================
feature_cols = [
    "pv_min", "pv_max", "pv_avg", "pv_median",
    "fav_min", "fav_max", "fav_avg", "fav_median",
    "cart_min", "cart_max", "cart_avg", "cart_median"
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"下列必需特征缺失：{missing}")

X = df[feature_cols].copy()
y = df[TARGET].copy()

# 基础清洗：替换无穷，填充缺失
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# =============================
# 4) 定义模型（Stacking: 基模型 XGB + RF + LR， 元学习器 LR）
#    注意：LR 作为基模型使用标准化
# =============================
# 基学习器
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, eval_metric="logloss")

# 对于基学习器里的 LR，配合 StandardScaler
lr_base = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(random_state=42, max_iter=1000))
])

# 元学习器（最终融合器）
meta_lr = LogisticRegression(random_state=42, max_iter=1000)

stack_model = StackingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("lr", lr_base)],
    final_estimator=meta_lr,
    stack_method="predict_proba",   # 使用基学习器的概率作为元学习器输入
    passthrough=False,
    cv=5
)

# =============================
# 5) 5折交叉验证（训练集内做 SMOTE）
# =============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accs, f1s, aucs, recalls = [], [], [], []
for tr_idx, va_idx in cv.split(X, y):
    X_tr, X_va = X.iloc[tr_idx].values, X.iloc[va_idx].values
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 仅在训练折上做 SMOTE
    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

    # 训练
    stack_model.fit(X_tr_res, y_tr_res)

    # 预测
    y_pred = stack_model.predict(X_va)
    y_proba = stack_model.predict_proba(X_va)[:, 1]

    # 指标
    accs.append(accuracy_score(y_va, y_pred))
    f1s.append(f1_score(y_va, y_pred))
    aucs.append(roc_auc_score(y_va, y_proba))
    recalls.append(recall_score(y_va, y_pred))

# =============================
# 6) 打印结果
# =============================
print("\n=== Stacking + SMOTE + XGB + LR + RF ===")
print(f"Accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
print(f"F1-Score: {np.mean(f1s):.4f} (+/- {np.std(f1s):.4f})")
print(f"AUC:      {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
print(f"Recall:   {np.mean(recalls):.4f} (+/- {np.std(recalls):.4f})")
