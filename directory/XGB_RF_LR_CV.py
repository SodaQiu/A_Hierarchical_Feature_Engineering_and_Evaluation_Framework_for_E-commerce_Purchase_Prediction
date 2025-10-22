import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from scipy.stats import randint, uniform

# =========================
# 1) 读取与准备数据
# =========================
file_path = r"C:\Users\47556\Desktop\no_day1.xlsx"
TARGET = "buy_yn"

df = pd.read_excel(file_path)

if TARGET not in df.columns:
    raise ValueError(f"未找到目标列 {TARGET}")

# 根据你的场景，删除无关/潜在泄漏列（按需调整）
cols_to_drop = [TARGET, "user_id", "date", "buy_min", "buy_max", "buy_avg"]
X = df.drop(columns=cols_to_drop, errors="ignore").copy()
y = df[TARGET].copy()

# 数值清理
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# 若还有非数值列，可在此处做独热：
# X = pd.get_dummies(X, drop_first=True)

# =========================
# 2) 定义基模型 & 参数空间
# =========================
# 2.1 Logistic Regression（带标准化）
pipe_lr = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])
param_lr = {
    "lr__C": uniform(0.01, 5.0),        # 0.01 ~ 5.01
    "lr__penalty": ["l2"],
    "lr__solver": ["lbfgs", "saga"]
    # 如需处理类不平衡，可加: "lr__class_weight": [None, "balanced"]
}

# 2.2 Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_rf = {
    "n_estimators": randint(120, 320),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None]
    # 如需处理类不平衡，可加: "class_weight": [None, "balanced", "balanced_subsample"]
}

# 2.3 XGBoost（去掉 use_label_encoder，hist 加速，静音）
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    tree_method="hist",   # 如果有GPU: 改为 "gpu_hist"
    n_jobs=-1,
    verbosity=0
)
param_xgb = {
    "n_estimators": randint(120, 320),
    "max_depth": randint(3, 9),
    "learning_rate": uniform(0.02, 0.18),   # 0.02~0.20
    "subsample": uniform(0.7, 0.3),         # 0.7~1.0
    "colsample_bytree": uniform(0.7, 0.3),  # 0.7~1.0
    "min_child_weight": randint(1, 6),
    "gamma": uniform(0.0, 0.4)
}

# =========================
# 3) 轻量级随机搜索（先粗调）
# =========================
def tune_model(model, param_dist, X, y, name, n_iter=12, inner_cv=3, scoring="f1"):
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,      # 可改为 "roc_auc" 或自定义
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
        verbose=0,
        error_score="raise"
    )
    search.fit(X, y)
    print(f"[{name}] Best params: {search.best_params_}")
    return search.best_estimator_

best_lr  = tune_model(pipe_lr, param_lr, X, y, name="LR")
best_rf  = tune_model(rf,       param_rf, X, y, name="RF")
best_xgb = tune_model(xgb,     param_xgb, X, y, name="XGB")

# =========================
# 4) 构建 Soft Voting 集成
# =========================
voting_soft = VotingClassifier(
    estimators=[("lr", best_lr), ("rf", best_rf), ("xgb", best_xgb)],
    voting="soft",
    n_jobs=-1
)

# =========================
# 5) 外层 5 折交叉验证评估
# =========================
def evaluate_cv(estimator, X, y, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, aucs, recalls = [], [], [], []
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        estimator.fit(X_tr, y_tr)
        y_pred  = estimator.predict(X_va)
        y_proba = estimator.predict_proba(X_va)[:, 1]

        accs.append(accuracy_score(y_va, y_pred))
        f1s.append(f1_score(y_va, y_pred))
        aucs.append(roc_auc_score(y_va, y_proba))
        recalls.append(recall_score(y_va, y_pred))

    return (
        np.mean(accs), np.std(accs),
        np.mean(f1s), np.std(f1s),
        np.mean(aucs), np.std(aucs),
        np.mean(recalls), np.std(recalls),
    )

models = {
    "Tuned LR (Pipeline)": best_lr,
    "Tuned RF": best_rf,
    "Tuned XGB": best_xgb,
    "Soft Voting (LR+RF+XGB)": voting_soft
}

print("\n=== 外层5折交叉验证结果 ===")
print("{:<28} {:<20} {:<20} {:<20} {:<20}".format("Model", "Accuracy", "F1", "AUC", "Recall"))
for name, est in models.items():
    acc_m, acc_s, f1_m, f1_s, auc_m, auc_s, rec_m, rec_s = evaluate_cv(est, X, y, n_splits=5, seed=42)
    print("{:<28} {:.4f} (+/-{:.4f})   {:.4f} (+/-{:.4f})   {:.4f} (+/-{:.4f})   {:.4f} (+/-{:.4f})".format(
        name, acc_m, acc_s, f1_m, f1_s, auc_m, auc_s, rec_m, rec_s
    ))
