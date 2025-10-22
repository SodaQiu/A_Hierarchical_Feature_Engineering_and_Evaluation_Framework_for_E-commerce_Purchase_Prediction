import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, recall_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


import matplotlib
matplotlib.use("TkAgg")   # 或者 "Qt5Agg"

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    df = df.fillna(0)

    if 'buy' in df.columns:
        df['buy_target'] = (df['buy'] > 0).astype(int)
    elif 'buy_target' not in df.columns:
        raise ValueError("没有找到 'buy' 或 'buy_target' 列")

    return df

def prepare_original_features(df):
    """只保留原始特征（不做特征工程）"""
    target_col = 'buy_target'
    exclude_cols = [target_col, 'user_id']

    # 移除 buy 相关的原始特征，避免数据泄露
    leakage_cols = ['buy', 'buy_min', 'buy_max', 'buy_avg', 'buy_yn']

    exclude_cols.extend([col for col in leakage_cols if col in df.columns])

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    processed_df = df[['user_id', target_col] + feature_cols].copy()

    # 编码分类变量
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['user_id', target_col]:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))

    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    processed_df = processed_df.fillna(0)



    return processed_df, feature_cols, target_col

def plot_all_feature_importance(all_feature_importance, all_selected_features):
    """绘制Top 20特征的重要性"""
    feature_importance_dict = {}
    for importance_scores, feature_names in zip(all_feature_importance, all_selected_features):
        for feature_name, importance in zip(feature_names, importance_scores):
            feature_importance_dict.setdefault(feature_name, []).append(importance)

    mean_importance = {f: np.mean(v) for f, v in feature_importance_dict.items()}
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)

    # 获取Top 20特征
    top_20_features = sorted_features[:20]
    top20 = pd.DataFrame(top_20_features, columns=['Feature', 'Importance'])

    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=top20, palette="coolwarm", legend=False)
    plt.title("Top 20 Important Features (Logistic Regression - Base Features)", fontsize=14)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig('lr_base_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== Top 20 Feature Importance ===")
    for i, (feature_name, importance) in enumerate(top_20_features, 1):
        print(f"{i:2d}. {feature_name:<30} {importance:.6f}")

def run_logistic_regression_original(file_path):
    df = load_and_preprocess_data(file_path)
    processed_df, feature_cols, target_col = prepare_original_features(df)

    X = processed_df[feature_cols].values
    y = processed_df[target_col].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_acc, all_f1, all_auc, all_pr_auc = [], [], [], []
    all_feature_importance, all_selected_features = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        model.fit(X_train_smote, y_train_smote)

        feature_importance = np.abs(model.coef_[0])
        all_feature_importance.append(feature_importance)
        all_selected_features.append(feature_cols)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        recall = recall_score(y_val, y_pred)

        all_acc.append(acc); all_f1.append(f1); all_auc.append(auc); all_pr_auc.append(pr_auc)

        print(f"[Fold {fold+1}] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Recall={recall:.4f}")

        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.show()

    print("\n=== 5-fold CV results (Original Features, No RFE) ===")
    print(f"Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
    #print(f"PR-AUC: {np.mean(all_pr_auc):.4f} ± {np.std(all_pr_auc):.4f}")
    print(f"Recall: {np.mean(recall):.4f} ± {np.std(recall):.4f}")

    plot_all_feature_importance(all_feature_importance, all_selected_features)

if __name__ == "__main__":
    # file_path = r"E:\google\Lab\alibaba\no_day2.xlsx"
    file_path= r'C:\Users\47556\Desktop\no_day1.xlsx'
    print("=== Logistic Regression (base features) ===")
    run_logistic_regression_original(file_path)
