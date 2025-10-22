# -*- coding: utf-8 -*-
# 完整版：5折CV + OOF PR-AUC & 校准曲线 + 防泄露流程（SMOTE在训练折内）
import matplotlib
matplotlib.use('Agg')  # 非交互后端：保存图像到文件，不弹窗

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, recall_score, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# -----------------------------
# 数据加载与基础处理
# -----------------------------
def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    df = df.fillna(0)

    if 'buy' in df.columns:
        df['buy_target'] = (df['buy'] > 0).astype(int)
    elif 'buy_target' not in df.columns:
        raise ValueError("没有找到 'buy' 或 'buy_target' 列")

    return df


# -----------------------------
# 特征工程：添加转化率/稳定性/交互等（含泄露剔除）
# -----------------------------
def add_conversion_features(df):
    df = df.copy()

    # 基础转化率特征（基于平均值）
    df["cart_to_pv_rate"] = df["cart_avg"] / (df["pv_avg"] + 1e-6)
    df["fav_to_pv_rate"] = df["fav_avg"] / (df["pv_avg"] + 1e-6)
    df["buy_to_pv_rate"] = df["buy_avg"] / (df["pv_avg"] + 1e-6)

    # 购买转化率（与 buy 直接相关，后面会剔除）
    df["buy_to_cart_rate"] = df["buy_avg"] / (df["cart_avg"] + 1e-6)
    df["buy_to_fav_rate"] = df["buy_avg"] / (df["fav_avg"] + 1e-6)

    # 综合兴趣转化率
    df["intent_to_pv_rate"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)

    # 使用 pv_count 的转化率
    df["cart_to_pv_count_rate"] = df["cart_avg"] / (df["pv_count"] + 1e-6)
    df["fav_to_pv_count_rate"] = df["fav_avg"] / (df["pv_count"] + 1e-6)
    df["buy_to_pv_count_rate"] = df["buy_avg"] / (df["pv_count"] + 1e-6)

    # 行为变异性特征
    df["pv_range"] = df["pv_max"] - df["pv_min"]
    df["cart_range"] = df["cart_max"] - df["cart_min"]
    df["fav_range"] = df["fav_max"] - df["fav_min"]
    df["buy_range"] = df["buy_max"] - df["buy_min"]

    # 行为稳定性
    df["pv_stability"] = df["pv_range"] / (df["pv_avg"] + 1e-6)
    df["cart_stability"] = df["cart_range"] / (df["cart_avg"] + 1e-6)
    df["fav_stability"] = df["fav_range"] / (df["fav_avg"] + 1e-6)
    df["buy_stability"] = df["buy_range"] / (df["buy_avg"] + 1e-6)

    # 总体活跃度
    df["total_avg_activity"] = df["pv_avg"] + df["cart_avg"] + df["fav_avg"] + df["buy_avg"]
    df["total_max_activity"] = df["pv_max"] + df["cart_max"] + df["fav_max"] + df["buy_max"]

    # 行为偏好分布
    df["cart_ratio"] = df["cart_avg"] / (df["total_avg_activity"] + 1e-6)
    df["fav_ratio"] = df["fav_avg"] / (df["total_avg_activity"] + 1e-6)
    df["pv_ratio"] = df["pv_avg"] / (df["total_avg_activity"] + 1e-6)
    df["buy_ratio"] = df["buy_avg"] / (df["total_avg_activity"] + 1e-6)

    # 高级交互特征（与 buy 的交互项后续会剔除）
    df["pv_cart_interaction"] = df["pv_avg"] * df["cart_avg"]
    df["pv_fav_interaction"] = df["pv_avg"] * df["fav_avg"]
    df["cart_fav_interaction"] = df["cart_avg"] * df["fav_avg"]
    df["cart_buy_interaction"] = df["cart_avg"] * df["buy_avg"]
    df["fav_buy_interaction"] = df["fav_avg"] * df["buy_avg"]

    # 偏好强度特征
    df["fav_cart_preference"] = df["fav_avg"] / (df["cart_avg"] + 1e-6)
    df["intent_intensity"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)
    df["purchase_intensity"] = df["buy_avg"] / (df["cart_avg"] + df["fav_avg"] + 1e-6)

    # 峰值行为
    df["max_engagement"] = np.maximum.reduce([df["pv_max"], df["cart_max"], df["fav_max"]])
    df["peak_purchase_ratio"] = df["buy_max"] / (df["max_engagement"] + 1e-6)

    # 用户活跃分层
    df['activity_level'] = pd.cut(
        df['total_avg_activity'],
        bins=[0, 2, 8, 20, float('inf')],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # 比例平衡与主导
    df['balance_score'] = 1 - np.abs(df['cart_ratio'] - df['fav_ratio'])
    df['dominance_feature'] = np.where(
        df['pv_ratio'] > 0.7, 0,
        np.where(df['cart_ratio'] > df['fav_ratio'], 1, 2)
    )
    df['conversion_potential'] = (
            df['cart_to_pv_rate'] + df['fav_to_pv_rate'] - df['cart_to_pv_rate'] * df['fav_to_pv_rate']
    )

    # 一致性
    df['behavior_consistency'] = 1 / (1 + df['pv_stability'] + df['cart_stability'] + df['fav_stability'])
    df['purchase_consistency'] = 1 / (1 + df['buy_stability'])

    return df


def prepare_and_engineer_features(df):
    """准备特征并添加转化率特征；剔除直/间接 buy 泄露特征"""
    df = add_conversion_features(df)

    target_col = 'buy_target'
    exclude_cols = [target_col, 'user_id']

    # 移除可能导致数据泄露的特征（buy相关）
    leakage_cols = [
        'buy','buy_yn', 'buy_min', 'buy_max', 'buy_avg',
        'buy_to_cart_rate', 'buy_to_fav_rate', 'buy_to_pv_rate', 'buy_to_pv_count_rate',
        'cart_buy_interaction', 'fav_buy_interaction', 'purchase_intensity',
        'buy_range', 'buy_stability', 'purchase_consistency', 'peak_purchase_ratio',
        'total_avg_activity', 'total_max_activity', 'buy_ratio'
    ]
    for col in leakage_cols:
        if col in df.columns:
            exclude_cols.append(col)

    # 选择用于后续交互项生成的数值特征
    feature_cols_temp = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols_temp].select_dtypes(include=[np.number]).columns.tolist()

    # 过滤低方差特征（仅用于挑选交互项来源，非最终删列）
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)
    X_numeric = df[numeric_cols]
    _ = variance_selector.fit_transform(X_numeric)
    high_variance_features = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]

    # 限制交互项来源数量，避免维度爆炸
    max_features_for_interaction = min(12, len(high_variance_features))
    selected_for_interaction = high_variance_features[:max_features_for_interaction]

    # 生成2阶交互项（仅交互项，不含多项式平方）
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interaction = df[selected_for_interaction]
    X_poly = poly.fit_transform(X_interaction)
    feature_names = poly.get_feature_names_out(selected_for_interaction)

    # 只保留交互项（去掉原始列）
    original_feature_names = set(selected_for_interaction)
    interaction_features = [col in feature_names and (col not in original_feature_names) for col in feature_names]
    # 将交互特征添加到原始数据框
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    for col in feature_names:
        if col not in original_feature_names:
            df[col] = poly_df[col]

    # 最终特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    processed_df = df[['user_id', target_col] + feature_cols].copy()

    # 处理类别型（LabelEncoder）
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['user_id', target_col]:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))

    # 替换无穷大/NaN
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 简单异常值截断（IQR）
    numeric_feature_cols = [col for col in feature_cols if processed_df[col].dtype in ['int64', 'float64']]
    for col in numeric_feature_cols:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)

    print(f"特征工程完成！使用特征数: {len(feature_cols)}")
    return processed_df, feature_cols, target_col


# -----------------------------
# 可视化：Top-20 特征重要性（按LR系数绝对值）
# -----------------------------
def plot_feature_importance(all_feature_importance, all_selected_features):
    # 汇总每折的重要性
    feature_importance_dict = {}
    for importance_scores, feature_names in zip(all_feature_importance, all_selected_features):
        for feature_name, importance in zip(feature_names, importance_scores):
            feature_importance_dict.setdefault(feature_name, []).append(importance)

    # 计算均值
    mean_importance = {k: np.mean(v) for k, v in feature_importance_dict.items()}

    # 取前20
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
    top_20_features = sorted_features[:20]
    top20 = pd.DataFrame(top_20_features, columns=['Feature', 'Importance'])

    # 画图
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top20)
    plt.title("Top 20 Important Features (Logistic Regression)")
    plt.xlabel("Mean Absolute Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig('lr_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n=== Top 20 Feature Importance ===")
    for i, (feature_name, importance) in enumerate(top_20_features, 1):
        print(f"{i:2d}. {feature_name:<30} {importance:.6f}")


# -----------------------------
# 主流程：5折CV + OOF 图表
# -----------------------------
def run_logistic_regression(file_path):
    # 创建输出目录（可选）
    os.makedirs('../LR_Model/figures', exist_ok=True)

    df = load_and_preprocess_data(file_path)
    processed_df, feature_cols, target_col = prepare_and_engineer_features(df)

    X = processed_df[feature_cols].values
    y = processed_df[target_col].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_acc, all_f1, all_auc, all_pr_auc, all_recall = [], [], [], [], []
    all_feature_importance = []  # 每折的LR系数绝对值
    all_selected_features = []   # 每折RFE选中的特征名

    # OOF 收集器
    oof_y_true = []
    oof_y_proba = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # === 1) SMOTE（仅在训练折内）===
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train)

        # === 2) 标准化（在SMOTE后的训练集上fit）===
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_val_scaled = scaler.transform(X_val_raw)

        # === 3) RFE（仅在训练数据上fit）===
        base_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='liblinear', random_state=42)
        n_features_to_select = min(30, X_train_scaled.shape[1])  # 最多30个
        rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1)
        X_train_rfe = rfe.fit_transform(X_train_scaled, y_train_smote)
        X_val_rfe = rfe.transform(X_val_scaled)

        # 记录本折入模特征名
        selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
        all_selected_features.append(selected_feature_names)

        # === 4) 训练最终LR ===
        model = LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000, solver='liblinear', random_state=42
        )
        model.fit(X_train_rfe, y_train_smote)

        # 重要性（绝对系数）
        feature_importance = np.abs(model.coef_[0])
        all_feature_importance.append(feature_importance)

        # === 5) 验证集预测 ===
        y_pred = model.predict(X_val_rfe)
        y_proba = model.predict_proba(X_val_rfe)[:, 1]

        # 评估
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        recall = recall_score(y_val, y_pred)

        all_acc.append(acc); all_f1.append(f1); all_auc.append(auc); all_pr_auc.append(pr_auc); all_recall.append(recall)

        print(f"[Fold {fold}] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, Recall={recall:.4f}")

        # 混淆矩阵（保存为图像）
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f'figures/confusion_matrix_fold{fold}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 收集 OOF
        oof_y_true.append(y_val)
        oof_y_proba.append(y_proba)

    # ---- 汇总 ----
    print("\n=== 5-fold cross-validation results ===")
    print(f"Acc:    {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"F1:     {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"AUC:    {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
    print(f"PR-AUC: {np.mean(all_pr_auc):.4f} ± {np.std(all_pr_auc):.4f}")
    print(f"Recall: {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")

    # 拼接 OOF
    oof_y_true = np.concatenate(oof_y_true, axis=0)
    oof_y_proba = np.concatenate(oof_y_proba, axis=0)

    # ===== OOF PR 曲线 =====
    pr_precision, pr_recall, _ = precision_recall_curve(oof_y_true, oof_y_proba)
    pr_auc_oof = average_precision_score(oof_y_true, oof_y_proba)
    print(f"OOF PR-AUC: {pr_auc_oof:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(pr_recall, pr_precision, label=f'PR curve (AP = {pr_auc_oof:.3f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/pr_curve_oof.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== OOF 校准曲线 =====
    prob_true, prob_pred = calibration_curve(oof_y_true, oof_y_proba, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model (OOF)')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical probability in bin")
    plt.title("Calibration Curve (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/calibration_curve_oof.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== OOF 最优阈值（按 F1）=====
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (oof_y_proba >= thr).astype(int)
        f1 = f1_score(oof_y_true, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"Best OOF threshold by F1: {best_thr:.2f}, F1={best_f1:.4f}")

    # 特征重要性图
    plot_feature_importance(all_feature_importance, all_selected_features)
    print("\n图像已保存至 ./figures 目录（含PR曲线、校准曲线、各折混淆矩阵），特征重要性图保存为 lr_feature_importance.png")



if __name__ == "__main__":
    # TODO: 修改成你的数据路径
    file_path = r'C:\Users\47556\Desktop\no_day1.xlsx'
    print("=== Logistic Regression 模型训练和评估 ===")
    run_logistic_regression(file_path)
