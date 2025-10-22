import matplotlib
matplotlib.use('Agg')  # 非交互后端：保存图像到文件，不弹窗

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------
# 自定义 IQR 截尾变换器（折内拟合，防止泄露）
# ----------------------------
class IQRCapper:
    """在训练集上计算 IQR 边界，对训练集和验证集同时裁剪。"""
    def __init__(self, k=1.5):
        self.k = k
        self.bounds_ = {}

    def fit(self, X):
        self.bounds_ = {}
        X_ = pd.DataFrame(X).copy()
        for c in X_.columns:
            if np.issubdtype(X_[c].dtype, np.number):
                q1, q3 = X_[c].quantile(0.25), X_[c].quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - self.k * iqr, q3 + self.k * iqr
                self.bounds_[c] = (low, high)
        return self

    def transform(self, X):
        X_ = pd.DataFrame(X).copy()
        for c, (low, high) in self.bounds_.items():
            if c in X_.columns:
                X_[c] = X_[c].clip(lower=low, upper=high)
        return X_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------
# 1. 数据加载和预处理函数
# ----------------------------
def load_and_preprocess_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')

    if 'buy' in df.columns:
        df['buy_target'] = (df['buy'] > 0).astype(int)
    elif 'buy_target' not in df.columns:
        raise ValueError("没有找到 'buy' 或 'buy_target' 列")

    return df


# ----------------------------
# 2. 特征工程函数
# ----------------------------
def add_conversion_features(df):
    df = df.copy()
    df["cart_to_pv_rate"] = df["cart_avg"] / (df["pv_avg"] + 1e-6)
    df["fav_to_pv_rate"] = df["fav_avg"] / (df["pv_avg"] + 1e-6)
    df["buy_to_pv_rate"] = df["buy_avg"] / (df["pv_avg"] + 1e-6)
    df["buy_to_cart_rate"] = df["buy_avg"] / (df["cart_avg"] + 1e-6)
    df["buy_to_fav_rate"] = df["buy_avg"] / (df["fav_avg"] + 1e-6)
    df["intent_to_pv_rate"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)
    df["pv_range"] = df["pv_max"] - df["pv_min"]
    df["cart_range"] = df["cart_max"] - df["cart_min"]
    df["fav_range"] = df["fav_max"] - df["fav_min"]
    df["buy_range"] = df["buy_max"] - df["buy_min"]
    df["pv_stability"] = df["pv_range"] / (df["pv_avg"] + 1e-6)
    df["cart_stability"] = df["cart_range"] / (df["cart_avg"] + 1e-6)
    df["fav_stability"] = df["fav_range"] / (df["fav_avg"] + 1e-6)
    df["buy_stability"] = df["buy_range"] / (df["buy_avg"] + 1e-6)
    # 注意：为避免泄露，保留不含 buy 的 total 活跃度
    df["total_avg_activity"] = df["pv_avg"] + df["cart_avg"] + df["fav_avg"] + df["buy_avg"]
    df["cart_ratio"] = df["cart_avg"] / (df["total_avg_activity"] + 1e-6)
    df["fav_ratio"] = df["fav_avg"] / (df["total_avg_activity"] + 1e-6)
    df["pv_ratio"] = df["pv_avg"] / (df["total_avg_activity"] + 1e-6)
    df["buy_ratio"] = df["buy_avg"] / (df["total_avg_activity"] + 1e-6)
    df["behavior_consistency"] = 1 / (1 + df["pv_stability"] + df["cart_stability"] + df["fav_stability"])
    df["purchase_consistency"] = 1 / (1 + df["buy_stability"])
    return df


def prepare_and_engineer_features(df):
    df = add_conversion_features(df)
    target_col = 'buy_target'
    exclude_cols = [target_col, 'user_id']

    # 与 buy 强相关、可能泄露的特征排除
    leakage_cols = [
        'buy','buy_yn', 'buy_min', 'buy_max', 'buy_avg',
        'buy_to_cart_rate', 'buy_to_fav_rate', 'buy_to_pv_rate',
        'cart_buy_interaction', 'fav_buy_interaction', 'purchase_intensity',
        'buy_range', 'buy_stability', 'purchase_consistency',
        'total_avg_activity', 'buy_ratio'
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    return df


# ----------------------------
# 3. CatBoost + RFE 训练与评估（含特征重要性）
# ----------------------------
def run_catboost_with_optimization(df, target_variable, feature_cols):
    X, y = df[feature_cols].copy(), target_variable.copy()
    cv_acc, cv_f1, cv_auc, cv_recall = [], [], [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # (1) 折内 IQR 截尾（仅用训练折分位数）
        capper = IQRCapper(k=1.5)
        X_train = capper.fit_transform(X_train)
        X_val = capper.transform(X_val)

        # (2) 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # (3) SMOTE 仅训练集
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        # (4) RFE 特征选择（用 CatBoost 做基学习器）
        n_feats_to_select = min(30, X_train_balanced.shape[1])
        rfe = RFE(estimator=CatBoostClassifier(random_state=42, verbose=0),
                  n_features_to_select=n_feats_to_select, step=1)
        X_train_rfe = rfe.fit_transform(X_train_balanced, y_train_balanced)
        X_val_rfe = rfe.transform(X_val_scaled)

        # (5) CatBoost 训练
        cb_model = CatBoostClassifier(
            iterations=500,
            depth=4,
            learning_rate=0.05,
            random_state=42,
            verbose=0,
            early_stopping_rounds=50
        )
        cb_model.fit(X_train_rfe, y_train_balanced)

        # (6) 评估
        y_pred_proba = cb_model.predict_proba(X_val_rfe)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        recall = recall_score(y_val, y_pred)

        cv_acc.append(acc); cv_f1.append(f1); cv_auc.append(auc); cv_recall.append(recall)
        print(f"[Fold {fold+1}] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Recall={recall:.4f}")

    # ---- CV 汇总 ----
    print("\n=== 5-fold cross-validation results ===")
    print(f"Acc:    {np.nanmean(cv_acc):.4f} ± {np.nanstd(cv_acc):.4f}")
    print(f"F1:     {np.nanmean(cv_f1):.4f} ± {np.nanstd(cv_f1):.4f}")
    print(f"AUC:    {np.nanmean(cv_auc):.4f} ± {np.nanstd(cv_auc):.4f}")
    print(f"Recall: {np.nanmean(cv_recall):.4f} ± {np.nanstd(cv_recall):.4f}")

    # ----------------------------
    # 训练最终模型获取特征重要性（使用全量 X,y）
    # ----------------------------
    # 最终模型也按与你的流程一致：Scaler → SMOTE → RFE → CatBoost
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    n_feats_to_select = min(30, X_balanced.shape[1])
    rfe_final = RFE(estimator=CatBoostClassifier(random_state=42, verbose=0),
                    n_features_to_select=n_feats_to_select, step=1)
    X_rfe_final = rfe_final.fit_transform(X_balanced, y_balanced)

    cb_final = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )
    cb_final.fit(X_rfe_final, y_balanced)

    # 特征重要性（与被选中的特征一一对应）
    feature_importance = cb_final.get_feature_importance()
    selected_mask = rfe_final.support_
    selected_features = [feature_cols[i] for i, keep in enumerate(selected_mask) if keep]

    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    # Top20 绘图
    top20 = importance_df.head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top20, palette="coolwarm")
    plt.title("Top 20 Important Features (CatBoost + RFE)")
    plt.xlabel("Feature Importance"); plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig('catboost_feature_importance.png', dpi=300, bbox_inches='tight')
    # 不调用 plt.show()，Agg 后端已保存

    print("\n=== Top 20 Feature Importance ===")
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:<30} {row['Importance']:.6f}")

    return {
        'Accuracy': np.nanmean(cv_acc),
        'F1': np.nanmean(cv_f1),
        'AUC': np.nanmean(cv_auc),
        'Recall': np.nanmean(cv_recall)
    }


# ----------------------------
# 主程序入口
# ----------------------------
if __name__ == "__main__":
    file_path = r"C:\Users\47556\Desktop\no_day1.xlsx"
    df = load_and_preprocess_data(file_path)
    target_variable = df['buy_target']
    df = prepare_and_engineer_features(df)

    target_col = 'buy_target'
    exclude_cols = [target_col, 'user_id']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 按你原结构：全局将 object 列做 LabelEncoder（保留原逻辑）
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    print("=== CatBoost + RFE with IQRCapper (5-fold CV) ===")
    results = run_catboost_with_optimization(df, target_variable, feature_cols)
    print("\n最终结果:", results)
    print("特征重要性图已保存为: catboost_feature_importance_1.png")
