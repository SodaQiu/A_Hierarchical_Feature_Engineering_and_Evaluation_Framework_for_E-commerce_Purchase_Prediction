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
    """只添加基础转化率特征"""
    df = df.copy()

    # 基础转化率特征（基于平均值）
    df["cart_to_pv_rate"] = df["cart_avg"] / (df["pv_avg"] + 1e-6)
    df["fav_to_pv_rate"] = df["fav_avg"] / (df["pv_avg"] + 1e-6)

    # 综合兴趣转化率
    df["intent_to_pv_rate"] = (df["cart_avg"] + df["fav_avg"]) / (df["pv_avg"] + 1e-6)

    # 使用pv_count的转化率
    df["cart_to_pv_count_rate"] = df["cart_avg"] / (df["pv_count"] + 1e-6)
    df["fav_to_pv_count_rate"] = df["fav_avg"] / (df["pv_count"] + 1e-6)

    return df

def prepare_and_engineer_features(df):
    """准备特征并添加转化率特征"""
    # 添加转化率特征
    df = add_conversion_features(df)

    target_col = 'buy_target'
    exclude_cols = [target_col, 'user_id']

    # 移除可能导致数据泄露的特征（buy相关的原始特征）
    leakage_cols = ['buy', 'buy_min', 'buy_max', 'buy_avg']
    for col in leakage_cols:
        if col in df.columns:
            exclude_cols.append(col)

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
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=top20, palette="coolwarm", legend=False)
    plt.title("Top 20 Important Features (Logistic Regression - Conversion Only)", fontsize=14)
    plt.xlabel("Feature Importance (Mean Absolute Coefficient)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig('lr_conversion_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n=== Top 20 Feature Importance ===")
    for i, (feature_name, importance) in enumerate(top_20_features, 1):
        print(f"{i:2d}. {feature_name:<30} {importance:.6f}")

def run_logistic_regression(file_path):
    """运行逻辑回归模型"""
    # 加载和预处理数据
    df = load_and_preprocess_data(file_path)
    print(f"原始数据形状: {df.shape}")

    # 特征工程
    processed_df, feature_cols, target_col = prepare_and_engineer_features(df)
    print(f"处理后数据形状: {processed_df.shape}")

    # 准备特征和目标变量
    X = processed_df[feature_cols]
    y = processed_df[target_col]

    print(f"\n目标变量分布:")
    print(Counter(y))

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    # 存储结果
    all_acc, all_f1, all_auc, all_recall = [], [], [], []
    fold_results = []
    all_feature_importance = []
    all_selected_features = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1} ===")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # SMOTE过采样
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        print(f"SMOTE后训练集分布: {Counter(y_train_balanced)}")

        # 特征选择 - RFE
        lr_for_rfe = LogisticRegression(random_state=42, max_iter=1000)
        rfe = RFE(estimator=lr_for_rfe, n_features_to_select=min(20, len(feature_cols)), step=1)
        X_train_selected = rfe.fit_transform(X_train_balanced, y_train_balanced)
        X_val_selected = rfe.transform(X_val_scaled)

        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
        print(f"选择的特征数量: {len(selected_features)}")

        # 训练逻辑回归模型
        lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        lr_model.fit(X_train_selected, y_train_balanced)

        # 预测
        y_pred = lr_model.predict(X_val_selected)
        y_pred_proba = lr_model.predict_proba(X_val_selected)[:, 1]

        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        ap = average_precision_score(y_val, y_pred_proba)
        recall = recall_score(y_val, y_pred)

        all_acc.append(accuracy)
        all_f1.append(f1)
        all_auc.append(auc)
        all_recall.append(recall)

        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'average_precision': ap,
            'recall': recall
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Recall: {recall:.4f}")

        # 特征重要性（基于系数绝对值）
        feature_importance = np.abs(lr_model.coef_[0])
        all_feature_importance.append(feature_importance)
        all_selected_features.append(selected_features)

        # 混淆矩阵可视化
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Purchase', 'Purchase'],
                    yticklabels=['No Purchase', 'Purchase'])
        plt.title(f'Confusion Matrix - Fold {fold + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'lr_conversion_confusion_matrix_fold_{fold + 1}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 打印交叉验证结果
    results_df = pd.DataFrame(fold_results)
    print(f"\n=== 5折交叉验证结果 ===")
    for i, row in results_df.iterrows():
        print(f"[Fold {row['fold']}] Acc={row['accuracy']:.4f}, F1={row['f1_score']:.4f}, AUC={row['auc']:.4f}, Recall={row['recall']:.4f}")

    print(f"\n=== 5-fold CV results ===")
    print(f"Acc: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"F1: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    print(f"AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")



    # 绘制特征重要性
    plot_feature_importance(all_feature_importance, all_selected_features)

    return results_df

if __name__ == "__main__":
    file_path = r'C:\Users\47556\Desktop\no_day1.xlsx'
    results = run_logistic_regression(file_path)