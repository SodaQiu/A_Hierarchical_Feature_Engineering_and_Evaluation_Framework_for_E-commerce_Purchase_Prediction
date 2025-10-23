import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

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

    # 选择数值型特征进行多项式特征生成
    numeric_cols = df[feature_cols_temp].select_dtypes(include=[np.number]).columns.tolist()

    # 为了避免特征爆炸，我们只对最重要的特征生成交互项
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)  # 移除方差很小的特征
    X_numeric = df[numeric_cols]
    X_variance_filtered = variance_selector.fit_transform(X_numeric)
    high_variance_features = [numeric_cols[i] for i in variance_selector.get_support(indices=True)]

    # 限制用于生成交互特征的特征数量（避免维度爆炸）
    max_features_for_interaction = min(12, len(high_variance_features))  # 随机森林可以处理更多特征
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

    # 将交互特征添加到原始数据框 - 使用pd.concat避免DataFrame碎片化
    if interaction_features:
        interaction_df = poly_df[interaction_features]
        df = pd.concat([df, interaction_df], axis=1)

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

    print(f"特征工程完成！使用特征数: {len(feature_cols)}")

    return processed_df, feature_cols, target_col

def run_random_forest(file_path):
    df = load_and_preprocess_data(file_path)
    processed_df, feature_cols, target_col = prepare_and_engineer_features(df)

    X = processed_df[feature_cols].values
    y = processed_df[target_col].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_acc, all_f1, all_auc, all_recall = [], [], [], []
    all_feature_importance = []  # Store feature importance from each fold
    all_selected_features = []   # Store selected feature names from each fold

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # 使用RFE进行特征选择
        base_model = RandomForestClassifier(
            n_estimators=100,  # 用于RFE的基础模型使用较少的树
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        n_features_to_select = min(20, X_train_smote.shape[1])  # 选择最多30个特征
        rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1)
        X_train_rfe = rfe.fit_transform(X_train_smote, y_train_smote)
        X_val_rfe = rfe.transform(X_val)

        # 获取选中的特征名称
        selected_features_rfe = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
        all_selected_features.append(selected_features_rfe)

        # 最终的随机森林模型
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_rfe, y_train_smote)

        # 收集特征重要性（基于RFE选择的特征）
        feature_importance_dict = {}
        for i, feature_name in enumerate(selected_features_rfe):
            feature_importance_dict[feature_name] = model.feature_importances_[i]
        all_feature_importance.append(feature_importance_dict)

        y_pred = model.predict(X_val_rfe)
        y_proba = model.predict_proba(X_val_rfe)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        #pr_auc = average_precision_score(y_val, y_proba)
        recall = recall_score(y_val, y_pred)

        all_acc.append(acc)
        all_f1.append(f1)
        all_auc.append(auc)
        all_recall.append(recall)

        print(f"[Fold {fold+1}] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Recall={recall:.4f}")

        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f'rf_confusion_matrix_fold_{fold+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("\n=== 5-fold cross-validation results ===")
    print(f"Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
    print(f"Recall: {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")

    # --- RFE特征选择统计 ---
    from collections import Counter
    all_features_flat = [feature for fold_features in all_selected_features for feature in fold_features]
    feature_selection_count = Counter(all_features_flat)


    # --- 计算平均特征重要性 ---
    # 收集所有特征的重要性
    all_features_importance = {}
    for fold_importance in all_feature_importance:
        for feature, importance in fold_importance.items():
            if feature not in all_features_importance:
                all_features_importance[feature] = []
            all_features_importance[feature].append(importance)

    # 计算平均重要性
    avg_importance = {}
    for feature, importances in all_features_importance.items():
        avg_importance[feature] = np.mean(importances)

    # 创建重要性DataFrame
    importance_df = pd.DataFrame([
        {"Feature": feature, "Importance": importance, "Selection_Count": feature_selection_count[feature]}
        for feature, importance in avg_importance.items()
    ]).sort_values("Importance", ascending=False)

    top20 = importance_df.head(20)

    # 特征重要性可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=top20, palette="coolwarm", legend=False)
    plt.title("Top 20 Important Features (Random Forest)", fontsize=14)
    plt.xlabel("Feature Importance (Mean Importance)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n=== Top 20 Feature Importance (with RFE selection count) ===")
    for idx, row in top20.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f} (selected in {row['Selection_Count']}/5 folds)")



if __name__ == "__main__":
    file_path =  r"C:\Users\47556\Desktop\no_day1.xlsx"
    print("=== Random Forest Model with RFE Training and Evaluation ===")
    run_random_forest(file_path)

