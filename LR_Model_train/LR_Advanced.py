# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix, recall_score
# from sklearn.linear_model import LogisticRegression
# import seaborn as sns
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
# from sklearn.feature_selection import RFE
#
# import matplotlib
# matplotlib.use('Agg') # <-- 添加此行，切换到非交互式后端
#
# import pandas as pd
# import numpy as np
# # ... 其他 import ...
#
#
# def load_and_preprocess_data(file_path):
#     """加载和预处理数据"""
#     df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
#     df = df.fillna(0)
#
#     if 'buy' in df.columns:
#         df['buy_target'] = (df['buy'] > 0).astype(int)
#     elif 'buy_target' not in df.columns:
#         raise ValueError("没有找到 'buy' 或 'buy_target' 列")
#
#     return df
#
#     # if 'buy_count' in df.columns:
#     #     df['buy_target'] = (df['buy_count'] > 0).astype(int)
#     # elif 'buy_target' not in df.columns:
#     #     raise ValueError("没有找到 'buy' 或 'buy_target' 列")
#     #
#     # return df
#
# def add_conversion_features(df):
#     """基于新特征列表添加转化率和交互特征"""
#     df = df.copy()
#
#     # 基础转化率特征（基于平均值）
#     df["cart_to_pv_rate"] = df["cart_avg"] / (df["pv_avg"] + 1e-6)
#     df["fav_to_pv_rate"] = df["fav_avg"] / (df["pv_avg"] + 1e-6)
#     df["buy_to_pv_rate"] = df["buy_avg"] / (df["pv_avg"] + 1e-6)
#
#     # 购买转化率
#     df["buy_to_cart_rate"] = df["buy_avg"] / (df["cart_avg"] + 1e-6)
#     df["buy_to_fav_rate"] = df["buy_avg"] / (df["fav_avg"] + 1e-6)
#
#     # 综合兴趣转化率
#     df["intent_to_pv_rate"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)
#
#     # 使用pv_count的转化率
#     df["cart_to_pv_count_rate"] = df["cart_avg"] / (df["pv_count"] + 1e-6)
#     df["fav_to_pv_count_rate"] = df["fav_avg"] / (df["pv_count"] + 1e-6)
#     df["buy_to_pv_count_rate"] = df["buy_avg"] / (df["pv_count"] + 1e-6)
#
#     # 行为变异性特征（基于min、max、avg）
#     df["pv_range"] = df["pv_max"] - df["pv_min"]
#     df["cart_range"] = df["cart_max"] - df["cart_min"]
#     df["fav_range"] = df["fav_max"] - df["fav_min"]
#     df["buy_range"] = df["buy_max"] - df["buy_min"]
#
#     # 行为稳定性（变异系数）
#     df["pv_stability"] = df["pv_range"] / (df["pv_avg"] + 1e-6)
#     df["cart_stability"] = df["cart_range"] / (df["cart_avg"] + 1e-6)
#     df["fav_stability"] = df["fav_range"] / (df["fav_avg"] + 1e-6)
#     df["buy_stability"] = df["buy_range"] / (df["buy_avg"] + 1e-6)
#
#     # 总体活跃度
#     df["total_avg_activity"] = df["pv_avg"] + df["cart_avg"] + df["fav_avg"] + df["buy_avg"]
#     df["total_max_activity"] = df["pv_max"] + df["cart_max"] + df["fav_max"] + df["buy_max"]
#
#     # 行为偏好分布（基于平均值）
#     df["cart_ratio"] = df["cart_avg"] / (df["total_avg_activity"] + 1e-6)
#     df["fav_ratio"] = df["fav_avg"] / (df["total_avg_activity"] + 1e-6)
#     df["pv_ratio"] = df["pv_avg"] / (df["total_avg_activity"] + 1e-6)
#     df["buy_ratio"] = df["buy_avg"] / (df["total_avg_activity"] + 1e-6)
#
#     # 高级交互特征
#     df["pv_cart_interaction"] = df["pv_avg"] * df["cart_avg"]
#     df["pv_fav_interaction"] = df["pv_avg"] * df["fav_avg"]
#     df["cart_fav_interaction"] = df["cart_avg"] * df["fav_avg"]
#     df["cart_buy_interaction"] = df["cart_avg"] * df["buy_avg"]
#     df["fav_buy_interaction"] = df["fav_avg"] * df["buy_avg"]
#
#     # 偏好强度特征
#     df["fav_cart_preference"] = df["fav_avg"] / (df["cart_avg"] + 1e-6)
#     df["intent_intensity"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)
#     df["purchase_intensity"] = df["buy_avg"] / (df["cart_avg"] + df["fav_avg"] + 1e-6)
#
#     # 峰值行为特征
#     df["max_engagement"] = np.maximum.reduce([df["pv_max"], df["cart_max"], df["fav_max"]])
#     df["peak_purchase_ratio"] = df["buy_max"] / (df["max_engagement"] + 1e-6)
#
#     # 用户活跃度分层（基于平均活跃度）
#     df['activity_level'] = pd.cut(
#         df['total_avg_activity'],
#         bins=[0, 2, 8, 20, float('inf')],
#         labels=[0, 1, 2, 3],
#         include_lowest=True
#     ).astype(int)
#
#     # 比例平衡特征
#     df['balance_score'] = 1 - np.abs(df['cart_ratio'] - df['fav_ratio'])
#     df['dominance_feature'] = np.where(
#         df['pv_ratio'] > 0.7, 0,  # 浏览主导
#         np.where(df['cart_ratio'] > df['fav_ratio'], 1, 2)  # 加购主导 vs 收藏主导
#     )
#     df['conversion_potential'] = df['cart_to_pv_rate'] + df['fav_to_pv_rate'] - df['cart_to_pv_rate'] * df['fav_to_pv_rate']
#
#     # 行为一致性特征（基于稳定性）
#     df['behavior_consistency'] = 1 / (1 + df['pv_stability'] + df['cart_stability'] + df['fav_stability'])
#     df['purchase_consistency'] = 1 / (1 + df['buy_stability'])
#
#     return df
#
# def prepare_and_engineer_features(df):
#     """准备特征并添加转化率特征"""
#     # 添加转化率特征
#     df = add_conversion_features(df)
#
#     target_col = 'buy_target'
#     exclude_cols = [target_col, 'user_id']
#
#     # 移除可能导致数据泄露的特征（buy相关的原始特征和转化率特征）
#     leakage_cols = ['buy', 'buy_yn', 'buy_min', 'buy_max', 'buy_avg',
#                     'buy_to_cart_rate', 'buy_to_fav_rate', 'buy_to_pv_rate', 'buy_to_pv_count_rate',
#                     'cart_buy_interaction', 'fav_buy_interaction', 'purchase_intensity',
#                     'buy_range', 'buy_stability', 'purchase_consistency', 'peak_purchase_ratio',
#                     'total_avg_activity', 'total_max_activity', 'buy_ratio']
#     for col in leakage_cols:
#         if col in df.columns:
#             exclude_cols.append(col)
#
#     # 生成高阶交互特征
#     feature_cols_temp = [col for col in df.columns if col not in exclude_cols]
#
#     # 选择数值型特征进行多项式特征生成
#     numeric_cols = df[feature_cols_temp].select_dtypes(include=[np.number]).columns.tolist()
#
#     # 为了避免特征爆炸，我们只对最重要的特征生成交互项
#     from sklearn.feature_selection import VarianceThreshold
#     variance_selector = VarianceThreshold(threshold=0.01)  # 移除方差很小的特征
#     X_numeric = df[numeric_cols]
#     X_variance_filtered = variance_selector.fit_transform(X_numeric)
#     high_variance_features = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]
#
#     # 限制用于生成交互特征的特征数量（避免维度爆炸）
#     max_features_for_interaction = min(12, len(high_variance_features))  # LR用较少特征避免过拟合
#     selected_for_interaction = high_variance_features[:max_features_for_interaction]
#
#     # 生成2阶多项式特征（包括交互项）
#     from sklearn.preprocessing import PolynomialFeatures
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_interaction = df[selected_for_interaction]
#     X_poly = poly.fit_transform(X_interaction)
#
#     # 获取多项式特征的名称
#     feature_names = poly.get_feature_names_out(selected_for_interaction)
#
#     # 将多项式特征添加到数据框中
#     poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
#
#     # 移除原始特征（避免重复），只保留交互项
#     original_feature_names = set(selected_for_interaction)
#     interaction_features = [col for col in feature_names if col not in original_feature_names]
#
#     # 将交互特征添加到原始数据框
#     for col in interaction_features:
#         df[col] = poly_df[col]
#
#
#     # 选择特征列
#     feature_cols = [col for col in df.columns if col not in exclude_cols]
#     processed_df = df[['user_id', target_col] + feature_cols].copy()
#
#     # 处理分类变量
#     categorical_cols = processed_df.select_dtypes(include=['object']).columns
#     for col in categorical_cols:
#         if col not in ['user_id', target_col]:
#             le = LabelEncoder()
#             processed_df[col] = le.fit_transform(processed_df[col].astype(str))
#
#     # 处理无穷大和NaN值
#     processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
#     processed_df = processed_df.fillna(0)
#
#     # 异常值处理 - 使用IQR方法检测和处理异常值
#     numeric_feature_cols = [col for col in feature_cols if processed_df[col].dtype in ['int64', 'float64']]
#     plot_overall_iqr_boxplot(processed_df, numeric_feature_cols, save_path="../Project_alibaba/figures/iqr_boxplot_overall.png")
#
#     for col in numeric_feature_cols:
#         Q1 = processed_df[col].quantile(0.25)
#         Q3 = processed_df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#
#         # 使用截断方法处理异常值
#         processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
#
#     # 检查特征方差
#     feature_variances = processed_df[feature_cols].var()
#     low_variance_features = feature_variances[feature_variances < 0.01].index.tolist()
#
#     print(f"特征工程完成！使用特征数: {len(feature_cols)}")
#
#     return processed_df, feature_cols, target_col
#
#
# # 绘制箱图
# import os
#
# def plot_overall_iqr_boxplot(df, numeric_cols, save_path="figures/iqr_boxplot_overall.png"):
#     """绘制整体箱线图，把所有数值特征画在一张图里"""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#     plt.figure(figsize=(max(12, len(numeric_cols) * 0.3), 6))
#     sns.boxplot(data=df[numeric_cols], orient="h", palette="Set2", showfliers=True)
#     plt.title("Overall IQR Boxplot for Numeric Features", fontsize=14)
#     plt.xlabel("Feature Value")
#     plt.ylabel("Features")
#     plt.tight_layout()
#
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"整体箱线图已保存: {save_path}")
#
#
#
#
# def plot_feature_importance(all_feature_importance, all_selected_features):
#     """Plot top 20 feature importance from cross-validation results"""
#     # Calculate average feature importance across all folds
#     feature_importance_dict = {}
#
#     for fold_idx, (importance_scores, feature_names) in enumerate(zip(all_feature_importance, all_selected_features)):
#         for feature_name, importance in zip(feature_names, importance_scores):
#             if feature_name not in feature_importance_dict:
#                 feature_importance_dict[feature_name] = []
#             feature_importance_dict[feature_name].append(importance)
#
#     # Calculate mean importance for each feature
#     mean_importance = {}
#     for feature_name, importance_list in feature_importance_dict.items():
#         mean_importance[feature_name] = np.mean(importance_list)
#
#     # Sort features by importance and get top 20
#     sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
#     top_20_features = sorted_features[:20]
#
#     # Prepare data for plotting
#     top20 = pd.DataFrame(top_20_features, columns=['Feature', 'Importance'])
#
#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x="Importance", y="Feature", data=top20, palette="coolwarm")
#     plt.title("Top 20 Important Features (Logistic Regression)", fontsize=14)
#     plt.xlabel("Feature Importance (Mean Absolute Coefficient)")
#     plt.ylabel("Feature Name")
#     plt.tight_layout()
#     plt.savefig('lr_feature_importance.png', dpi=300, bbox_inches='tight')
#     # plt.show()
#
#     print(f"\n=== Top 20 Feature Importance ===")
#     for i, (feature_name, importance) in enumerate(top_20_features, 1):
#         print(f"{i:2d}. {feature_name:<30} {importance:.6f}")
#
# def run_logistic_regression(file_path):
#     df = load_and_preprocess_data(file_path)
#     processed_df, feature_cols, target_col = prepare_and_engineer_features(df)
#
#     X = processed_df[feature_cols].values
#     y = processed_df[target_col].values
#
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     all_acc, all_f1, all_auc, all_pr_auc, all_recall = [], [], [], [], []
#     all_feature_importance = []  # Store feature importance from each fold
#     all_selected_features = []   # Store selected feature names from each fold
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#
#         # 标准化处理
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_val = scaler.transform(X_val)
#
#         # 使用SMOTE处理类别不平衡
#         smote = SMOTE(random_state=42)
#         X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#
#         # 递归特征消除(RFE)进行特征选择
#         base_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='liblinear', random_state=42)
#         n_features_to_select = min(30, X_train_smote.shape[1])  # 选择最多30个特征
#         rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1)
#         X_train_rfe = rfe.fit_transform(X_train_smote, y_train_smote)
#         X_val_rfe = rfe.transform(X_val)
#
#         # 最终的逻辑回归模型
#         model = LogisticRegression(
#             penalty='l2',
#             C=1.0,
#             max_iter=1000,
#             solver='liblinear',
#             random_state=42
#
#             # penalty='l1',
#             # C=0.5,
#             # solver='liblinear',
#             # max_iter=500,
#             # random_state=42
#         )
#         model.fit(X_train_rfe, y_train_smote)
#
#         # 收集特征重要性 (逻辑回归系数的绝对值)
#         feature_importance = np.abs(model.coef_[0])
#         selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
#         all_feature_importance.append(feature_importance)
#         all_selected_features.append(selected_feature_names)
#
#         # 预测
#         y_pred = model.predict(X_val_rfe)
#         y_proba = model.predict_proba(X_val_rfe)[:, 1]
#
#         acc = accuracy_score(y_val, y_pred)
#         f1 = f1_score(y_val, y_pred)
#         auc = roc_auc_score(y_val, y_proba)
#         pr_auc = average_precision_score(y_val, y_proba)
#
#         all_acc.append(acc)
#         all_f1.append(f1)
#         all_auc.append(auc)
#         all_pr_auc.append(pr_auc)
#
#         # 计算recall
#         recall = recall_score(y_val, y_pred)
#         all_recall.append(recall)
#
#         print(f"[Fold {fold+1}] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, Recall={recall:.4f}")
#
#         cm = confusion_matrix(y_val, y_pred)
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#         plt.title(f"Confusion Matrix - Fold {fold+1}")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.show()
#
#     print("\n=== 5-fold cross-validation results ===")
#     print(f"Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
#     print(f"F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
#     print(f"AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
#     print(f"PR-AUC: {np.mean(all_pr_auc):.4f} ± {np.std(all_pr_auc):.4f}")
#     print(f"Recall: {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
#
#
#     # Plot feature importance
#     plot_feature_importance(all_feature_importance, all_selected_features)
#
# if __name__ == "__main__":
#
#     file_path= r'C:\Users\47556\Desktop\no_day1.xlsx'
#     print("=== Logistic Regression模型训练和评估 ===")
#     run_logistic_regression(file_path)


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix, recall_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from collections import Counter

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    df = df.fillna(0)

    if 'buy' in df.columns:
        df['buy_target'] = (df['buy'] > 0).astype(int)
    elif 'buy_target' not in df.columns:
        raise ValueError("没有找到 'buy' 或 'buy_target' 列")

    return df

def add_conversion_features(df):
    """基于新特征列表添加转化率和交互特征"""
    df = df.copy()

    # 基础转化率特征（基于平均值）
    df["cart_to_pv_rate"] = df["cart_avg"] / (df["pv_avg"] + 1e-6)
    df["fav_to_pv_rate"] = df["fav_avg"] / (df["pv_avg"] + 1e-6)
    df["buy_to_pv_rate"] = df["buy_avg"] / (df["pv_avg"] + 1e-6)

    # 购买转化率
    df["buy_to_cart_rate"] = df["buy_avg"] / (df["cart_avg"] + 1e-6)
    df["buy_to_fav_rate"] = df["buy_avg"] / (df["fav_avg"] + 1e-6)

    # 综合兴趣转化率
    df["intent_to_pv_rate"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)

    # 使用pv_count的转化率
    df["cart_to_pv_count_rate"] = df["cart_avg"] / (df["pv_count"] + 1e-6)
    df["fav_to_pv_count_rate"] = df["fav_avg"] / (df["pv_count"] + 1e-6)
    df["buy_to_pv_count_rate"] = df["buy_avg"] / (df["pv_count"] + 1e-6)

    # 行为变异性特征（基于min、max、avg）
    df["pv_range"] = df["pv_max"] - df["pv_min"]
    df["cart_range"] = df["cart_max"] - df["cart_min"]
    df["fav_range"] = df["fav_max"] - df["fav_min"]
    df["buy_range"] = df["buy_max"] - df["buy_min"]

    # 行为稳定性（变异系数）
    df["pv_stability"] = df["pv_range"] / (df["pv_avg"] + 1e-6)
    df["cart_stability"] = df["cart_range"] / (df["cart_avg"] + 1e-6)
    df["fav_stability"] = df["fav_range"] / (df["fav_avg"] + 1e-6)
    df["buy_stability"] = df["buy_range"] / (df["buy_avg"] + 1e-6)

    # 总体活跃度
    df["total_avg_activity"] = df["pv_avg"] + df["cart_avg"] + df["fav_avg"] + df["buy_avg"]
    df["total_max_activity"] = df["pv_max"] + df["cart_max"] + df["fav_max"] + df["buy_max"]

    # 行为偏好分布（基于平均值）
    df["cart_ratio"] = df["cart_avg"] / (df["total_avg_activity"] + 1e-6)
    df["fav_ratio"] = df["fav_avg"] / (df["total_avg_activity"] + 1e-6)
    df["pv_ratio"] = df["pv_avg"] / (df["total_avg_activity"] + 1e-6)
    df["buy_ratio"] = df["buy_avg"] / (df["total_avg_activity"] + 1e-6)

    # 高级交互特征
    df["pv_cart_interaction"] = df["pv_avg"] * df["cart_avg"]
    df["pv_fav_interaction"] = df["pv_avg"] * df["fav_avg"]
    df["cart_fav_interaction"] = df["cart_avg"] * df["fav_avg"]
    df["cart_buy_interaction"] = df["cart_avg"] * df["buy_avg"]
    df["fav_buy_interaction"] = df["fav_avg"] * df["buy_avg"]

    # 偏好强度特征
    df["fav_cart_preference"] = df["fav_avg"] / (df["cart_avg"] + 1e-6)
    df["intent_intensity"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)
    df["purchase_intensity"] = df["buy_avg"] / (df["cart_avg"] + df["fav_avg"] + 1e-6)

    # 峰值行为特征
    df["max_engagement"] = np.maximum.reduce([df["pv_max"], df["cart_max"], df["fav_max"]])
    df["peak_purchase_ratio"] = df["buy_max"] / (df["max_engagement"] + 1e-6)

    # 用户活跃度分层（基于平均活跃度）
    df['activity_level'] = pd.cut(
        df['total_avg_activity'],
        bins=[0, 2, 8, 20, float('inf')],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # 比例平衡特征
    df['balance_score'] = 1 - np.abs(df['cart_ratio'] - df['fav_ratio'])
    df['dominance_feature'] = np.where(
        df['pv_ratio'] > 0.7, 0,  # 浏览主导
        np.where(df['cart_ratio'] > df['fav_ratio'], 1, 2)  # 加购主导 vs 收藏主导
    )
    df['conversion_potential'] = df['cart_to_pv_rate'] + df['fav_to_pv_rate'] - df['cart_to_pv_rate'] * df['fav_to_pv_rate']

    # 行为一致性特征（基于稳定性）
    df['behavior_consistency'] = 1 / (1 + df['pv_stability'] + df['cart_stability'] + df['fav_stability'])
    df['purchase_consistency'] = 1 / (1 + df['buy_stability'])

    return df

def prepare_and_engineer_features(df):
    """准备特征并添加转化率特征"""
    # 添加转化率特征
    df = add_conversion_features(df)

    target_col = 'buy_target'
    exclude_cols = [target_col, 'user_id']

    # 移除可能导致数据泄露的特征（buy相关的原始特征和转化率特征）
    leakage_cols = ['buy','buy_yn', 'buy_min', 'buy_max', 'buy_avg',
                    'buy_to_cart_rate', 'buy_to_fav_rate', 'buy_to_pv_rate', 'buy_to_pv_count_rate',
                    'cart_buy_interaction', 'fav_buy_interaction', 'purchase_intensity',
                    'buy_range', 'buy_stability', 'purchase_consistency', 'peak_purchase_ratio',
                    'total_avg_activity', 'total_max_activity', 'buy_ratio']
    for col in leakage_cols:
        if col in df.columns:
            exclude_cols.append(col)

    # 生成高阶交互特征
    feature_cols_temp = [col for col in df.columns if col not in exclude_cols]

    # 选择数值型特征进行多项式特征生成
    numeric_cols = df[feature_cols_temp].select_dtypes(include=[np.number]).columns.tolist()

    # 为了避免特征爆炸，我们只对最重要的特征生成交互项
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)  # 移除方差很小的特征
    X_numeric = df[numeric_cols]
    X_variance_filtered = variance_selector.fit_transform(X_numeric)
    high_variance_features = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]

    # 限制用于生成交互特征的特征数量（避免维度爆炸）
    max_features_for_interaction = min(12, len(high_variance_features))  # LR用较少特征避免过拟合
    selected_for_interaction = high_variance_features[:max_features_for_interaction]

    # 生成2阶多项式特征（包括交互项）
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interaction = df[selected_for_interaction]
    X_poly = poly.fit_transform(X_interaction)

    # 获取多项式特征的名称
    feature_names = poly.get_feature_names_out(selected_for_interaction)

    # 将多项式特征添加到数据框中
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)

    # 移除原始特征（避免重复），只保留交互项
    original_feature_names = set(selected_for_interaction)
    interaction_features = [col for col in feature_names if col not in original_feature_names]

    # 将交互特征添加到原始数据框
    for col in interaction_features:
        df[col] = poly_df[col]

    # 选择特征列
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    processed_df = df[['user_id', target_col] + feature_cols].copy()

    # 处理分类变量
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['user_id', target_col]:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))

    # 处理无穷大和NaN值
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    processed_df = processed_df.fillna(0)

    # 异常值处理 - 使用IQR方法检测和处理异常值
    numeric_feature_cols = [col for col in feature_cols if processed_df[col].dtype in ['int64', 'float64']]

    for col in numeric_feature_cols:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 使用截断方法处理异常值
        processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)

    # 检查特征方差
    feature_variances = processed_df[feature_cols].var()
    low_variance_features = feature_variances[feature_variances < 0.01].index.tolist()

    print(f"特征工程完成！使用特征数: {len(feature_cols)}")

    return processed_df, feature_cols, target_col

def plot_feature_importance(all_feature_importance, all_selected_features):
    """Plot top 20 feature importance from cross-validation results"""
    # Calculate average feature importance across all folds
    feature_importance_dict = {}

    for fold_idx, (importance_scores, feature_names) in enumerate(zip(all_feature_importance, all_selected_features)):
        for feature_name, importance in zip(feature_names, importance_scores):
            if feature_name not in feature_importance_dict:
                feature_importance_dict[feature_name] = []
            feature_importance_dict[feature_name].append(importance)

    # Calculate mean importance for each feature
    mean_importance = {}
    for feature_name, importance_list in feature_importance_dict.items():
        mean_importance[feature_name] = np.mean(importance_list)

    # Sort features by importance and get top 20
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
    top_20_features = sorted_features[:20]

    # Prepare data for plotting
    top20 = pd.DataFrame(top_20_features, columns=['Feature', 'Importance'])

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top20, palette="coolwarm")
    plt.title("Top 20 Important Features (Logistic Regression)", fontsize=14)
    plt.xlabel("Feature Importance (Mean Absolute Coefficient)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig('lr_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== Top 20 Feature Importance ===")
    for i, (feature_name, importance) in enumerate(top_20_features, 1):
        print(f"{i:2d}. {feature_name:<30} {importance:.6f}")

def run_logistic_regression(file_path):
    df = load_and_preprocess_data(file_path)
    processed_df, feature_cols, target_col = prepare_and_engineer_features(df)

    X = processed_df[feature_cols].values
    y = processed_df[target_col].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_acc, all_f1, all_auc, all_pr_auc, all_recall = [], [], [], [], []
    all_feature_importance = []  # Store feature importance from each fold
    all_selected_features = []   # Store selected feature names from each fold

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化处理
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 使用SMOTE处理类别不平衡
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # 递归特征消除(RFE)进行特征选择
        base_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='liblinear', random_state=42)
        n_features_to_select = min(30, X_train_smote.shape[1])  # 选择最多30个特征
        rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1)
        X_train_rfe = rfe.fit_transform(X_train_smote, y_train_smote)
        X_val_rfe = rfe.transform(X_val)

        # 最终的逻辑回归模型
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            solver='liblinear',
            random_state=42
        )
        model.fit(X_train_rfe, y_train_smote)

        # 收集特征重要性 (逻辑回归系数的绝对值)
        feature_importance = np.abs(model.coef_[0])
        selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
        all_feature_importance.append(feature_importance)
        all_selected_features.append(selected_feature_names)

        # 预测
        y_pred = model.predict(X_val_rfe)
        y_proba = model.predict_proba(X_val_rfe)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)

        all_acc.append(acc)
        all_f1.append(f1)
        all_auc.append(auc)
        all_pr_auc.append(pr_auc)

        # 计算recall
        recall = recall_score(y_val, y_pred)
        all_recall.append(recall)

        print(f"[Fold {fold+1}] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, Recall={recall:.4f}")

        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    print("\n=== 5-fold cross-validation results ===")
    print(f"Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
    print(f"PR-AUC: {np.mean(all_pr_auc):.4f} ± {np.std(all_pr_auc):.4f}")
    print(f"Recall: {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")

    # Plot feature importance
    plot_feature_importance(all_feature_importance, all_selected_features)

if __name__ == "__main__":
    file_path = r'C:\Users\47556\Desktop\no_day1.xlsx'  # 更新数据路径

    print("=== Logistic Regression模型训练和评估 ===")
    run_logistic_regression(file_path)
