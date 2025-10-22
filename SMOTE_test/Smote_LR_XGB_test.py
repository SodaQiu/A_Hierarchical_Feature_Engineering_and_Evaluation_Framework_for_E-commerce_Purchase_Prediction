from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA # 引入PCA用于可视化
from xgboost import XGBClassifier   # 新增 XGB

import matplotlib
matplotlib.use('Agg') # 切换到非交互式后端，防止在服务器上运行时出错

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
import sys, io, os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# ============= 数据预处理与特征工程（保持不变） =============
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    df = df.fillna(0)

    if 'buy' in df.columns:
        df['buy_target'] = (df['buy'] > 0).astype(int)
    elif 'buy_target' not in df.columns:
        raise ValueError("没有找到 'buy' 或 'buy_target' 列")
    return df

def add_conversion_features(df):
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
    df = add_conversion_features(df)

    target_col = 'buy_target'
    exclude_cols = [target_col, 'user_id']

    # 移除可能导致数据泄露的特征
    leakage_cols = ['buy', 'buy_yn', 'buy_min', 'buy_max', 'buy_avg',
                    'buy_to_cart_rate', 'buy_to_fav_rate', 'buy_to_pv_rate', 'buy_to_pv_count_rate',
                    'cart_buy_interaction', 'fav_buy_interaction', 'purchase_intensity',
                    'buy_range', 'buy_stability', 'purchase_consistency', 'peak_purchase_ratio',
                    'total_avg_activity', 'total_max_activity', 'buy_ratio']
    for col in leakage_cols:
        if col in df.columns:
            exclude_cols.append(col)

    # 生成高阶交互特征
    feature_cols_temp = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols_temp].select_dtypes(include=[np.number]).columns.tolist()

    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)
    X_numeric = df[numeric_cols]
    X_variance_filtered = variance_selector.fit_transform(X_numeric)
    high_variance_features = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]

    max_features_for_interaction = min(12, len(high_variance_features))
    selected_for_interaction = high_variance_features[:max_features_for_interaction]

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interaction = df[selected_for_interaction]
    X_poly = poly.fit_transform(X_interaction)
    feature_names = poly.get_feature_names_out(selected_for_interaction)
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)

    original_feature_names = set(selected_for_interaction)
    interaction_features = [col for col in feature_names if col not in original_feature_names]
    for col in interaction_features:
        df[col] = poly_df[col]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    processed_df = df[['user_id', target_col] + feature_cols].copy()

    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['user_id', target_col]:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))

    processed_df = processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)

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

def visualize_smote_effect(X_train, y_train, X_train_smote, y_train_smote):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_train_smote_pca = pca.transform(X_train_smote)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, ax=axes[0], alpha=0.7, palette='Set1')
    sns.scatterplot(x=X_train_smote_pca[:, 0], y=X_train_smote_pca[:, 1], hue=y_train_smote, ax=axes[1], alpha=0.7, palette='Set1')
    plt.suptitle('Data Distribution Before and After SMOTE (PCA Reduced)', fontsize=16)
    plt.savefig('smote_visualization.png', dpi=300)
    plt.close()
    print("SMOTE 可视化图已保存为 'smote_visualization.png'")


# ============= 模型训练与评估 =============
def train_and_evaluate(X, y, feature_cols, use_smote=True, model_type='LR'):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = {'acc': [], 'f1': [], 'auc': [], 'pr_auc': [], 'recall': []}
    all_feature_importance, all_selected_features = [], []
    is_visualized = False

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_final, y_train_final = X_train_scaled, y_train

        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            if not is_visualized:
                visualize_smote_effect(X_train_scaled, y_train, X_train_final, y_train_final)
                is_visualized = True

        if model_type == 'LR':
            base_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='liblinear', random_state=42)
            n_features_to_select = min(30, X_train_final.shape[1])
            rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1)
            X_train_rfe = rfe.fit_transform(X_train_final, y_train_final)
            X_val_rfe = rfe.transform(X_val_scaled)
            model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, solver='liblinear', random_state=42)
            model.fit(X_train_rfe, y_train_final)
            feature_importance = np.abs(model.coef_[0])
            selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]

        elif model_type == 'XGB':
            model = XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="logloss", use_label_encoder=False
            )
            model.fit(X_train_final, y_train_final)
            X_val_rfe = X_val_scaled
            feature_importance = model.feature_importances_
            selected_feature_names = feature_cols

        y_pred = model.predict(X_val_rfe)
        y_proba = model.predict_proba(X_val_rfe)[:, 1]

        all_metrics['acc'].append(accuracy_score(y_val, y_pred))
        all_metrics['f1'].append(f1_score(y_val, y_pred))
        all_metrics['auc'].append(roc_auc_score(y_val, y_proba))
        all_metrics['pr_auc'].append(average_precision_score(y_val, y_proba))
        all_metrics['recall'].append(recall_score(y_val, y_pred))

        all_feature_importance.append(feature_importance)
        all_selected_features.append(selected_feature_names)

    print(f"\n--- 评估结果 ({model_type}, {'With SMOTE' if use_smote else 'Without SMOTE'}) ---")
    for metric, scores in all_metrics.items():
        print(f"{metric.upper()}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return all_feature_importance, all_selected_features

# def plot_feature_importance(all_feature_importance, all_selected_features, ):
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

def plot_feature_importance(all_feature_importance, all_selected_features, model_name=''):
    """绘制最重要的20个特征"""
    feature_importance_dict = {}
    for importance_scores, feature_names in zip(all_feature_importance, all_selected_features):
        for feature_name, importance in zip(feature_names, importance_scores):
            feature_importance_dict.setdefault(feature_name, []).append(importance)

    mean_importance = {name: np.mean(scores) for name, scores in feature_importance_dict.items()}
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    top20 = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=top20, palette="viridis")
    plt.title(f"Top 20 Important Features ({model_name})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n=== Top 20 Feature Importance ({model_name}) ===")
    for i, (feature_name, importance) in enumerate(sorted_features, 1):
        print(f"{i:2d}. {feature_name:<30} {importance:.6f}")


# ============= 主流程 =============
def run_comparison_experiment(file_path):
    df = load_and_preprocess_data(file_path)
    processed_df, feature_cols, target_col = prepare_and_engineer_features(df)
    X = processed_df[feature_cols].values
    y = processed_df[target_col].values

    print("="*60)
    print("🚀 开始进行SMOTE对比实验 (LR & XGB) 🚀")
    print("="*60)

    for model_type in ['LR', 'XGB']:
        print(f"\n{model_type} 无SMOTE".center(60, "="))
        imp_no, feat_no = train_and_evaluate(X, y, feature_cols, use_smote=False, model_type=model_type)
        plot_feature_importance(imp_no, feat_no, f"{model_type}_Without_SMOTE")

        print(f"\n{model_type} 有SMOTE".center(60, "="))
        imp_yes, feat_yes = train_and_evaluate(X, y, feature_cols, use_smote=True, model_type=model_type)
        plot_feature_importance(imp_yes, feat_yes, f"{model_type}_With_SMOTE")

    print("\n🎉 对比实验完成，请查看输出与生成的图片 🎉")


if __name__ == "__main__":
    file_path = r'C:\Users\47556\Desktop\no_day1.xlsx'
    run_comparison_experiment(file_path)
