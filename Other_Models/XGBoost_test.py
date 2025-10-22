# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, recall_score
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline
#
#
# warnings.filterwarnings('ignore')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # ----------------------------
# # 1. 数据加载和预处理函数
# # ----------------------------
# def load_and_preprocess_data(file_path):
#     """加载数据并进行预处理"""
#     # 根据文件扩展名选择加载方式
#     if file_path.endswith('.xlsx'):
#         df = pd.read_excel(file_path)
#     else:
#         df = pd.read_csv(file_path)
#
#     # 缺失值处理
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     categorical_cols = df.select_dtypes(include=['object']).columns
#
#     # 数值型特征用中位数填充
#     for col in numeric_cols:
#         if df[col].isnull().sum() > 0:
#             median_value = df[col].median()
#             df[col] = df[col].fillna(median_value)
#
#     # 分类特征用众数填充
#     for col in categorical_cols:
#         df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
#
#     # 创建目标变量
#     if 'buy' in df.columns:
#         df['buy_target'] = (df['buy'] > 0).astype(int)
#     elif 'buy_target' in df.columns:
#         pass  # 已存在目标变量
#     else:
#         raise ValueError("没有找到 'buy' 或 'buy_target' 列")
#
#     return df
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
# # ----------------------------
# # 数据加载和预处理
# # ----------------------------
# print("=== 数据加载和预处理 ===")
# file_path = r"C:\Users\47556\Desktop\no_day1.xlsx"
# df = load_and_preprocess_data(file_path)
# target_variable = 'buy_target'
#
# # ----------------------------
# # 2. 特征工程和数据泄露防护
# # ----------------------------
# def prepare_and_engineer_features(df):
#     """准备特征并添加转化率特征"""
#     # 添加转化率特征
#     df = add_conversion_features(df)
#
#     target_col = 'buy_target'
#     exclude_cols = [target_col, 'user_id']
#
#     # 移除可能导致数据泄露的特征（buy相关的原始特征和转化率特征）
#     leakage_cols = ['buy','buy_yn', 'buy_min', 'buy_max', 'buy_avg',
#                     'buy_to_cart_rate', 'buy_to_fav_rate', 'buy_to_pv_rate', 'buy_to_pv_count_rate',
#                     'cart_buy_interaction', 'fav_buy_interaction', 'purchase_intensity',
#                     'buy_range', 'buy_stability', 'purchase_consistency', 'peak_purchase_ratio',
#                     'total_avg_activity', 'total_max_activity', 'buy_ratio']
#
#     for col in leakage_cols:
#         if col in df.columns:
#             df = df.drop(columns=[col])
#
#     # 生成高阶交互特征
#     # 选择数值型特征进行交互
#     numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
#
#     # 移除目标变量和排除列
#     feature_cols = [col for col in numeric_features if col not in exclude_cols + leakage_cols]
#
#     # 方差阈值过滤
#     from sklearn.feature_selection import VarianceThreshold
#     selector = VarianceThreshold(threshold=0.01)
#
#     # 先处理无穷大和NaN值
#     df_features = df[feature_cols].copy()
#     df_features = df_features.replace([np.inf, -np.inf], 0)
#     df_features = df_features.fillna(0)
#
#     # 应用方差过滤
#     try:
#         selector.fit(df_features)
#         selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
#     except:
#         selected_features = feature_cols
#
#     # 限制特征数量以避免过度复杂
#     if len(selected_features) > 20:
#         selected_features = selected_features[:20]
#
#     # 生成2阶多项式特征
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#
#     try:
#         poly_features = poly.fit_transform(df_features[selected_features])
#         poly_feature_names = poly.get_feature_names_out(selected_features)
#
#         # 移除原始特征，只保留交互项
#         interaction_names = [name for name in poly_feature_names if ' ' in name]
#         interaction_features = poly_features[:, [i for i, name in enumerate(poly_feature_names) if ' ' in name]]
#
#         # 添加交互特征到数据框
#         for i, name in enumerate(interaction_names[:50]):  # 限制交互特征数量
#             clean_name = name.replace(' ', '_x_')
#             df[f'interaction_{clean_name}'] = interaction_features[:, i]
#
#     except Exception as e:
#         pass  # 静默处理多项式特征生成失败
#
#     return df
#
# # 应用特征工程
# df = prepare_and_engineer_features(df)
#
# # 最终特征选择
# target_col = 'buy_target'
# exclude_cols = [target_col, 'user_id']
# feature_cols = [col for col in df.columns if col not in exclude_cols]
#
# # 处理分类变量编码
# le_dict = {}
# for col in df.columns:
#     if df[col].dtype == 'object':
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))
#         le_dict[col] = le
#
# # 处理无穷大和NaN值
# df = df.replace([np.inf, -np.inf], 0)
# df = df.fillna(0)
#
# # 异常值处理（IQR方法）
# for col in feature_cols:
#     if df[col].dtype in ['int64', 'float64']:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 2.5 * IQR
#         upper_bound = Q3 + 2.5 * IQR
#
#         # 截断异常值而不是删除
#         df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
#
# # 特征分布检查
# print(f"\n=== 特征工程完成 ===")
# print(f"最终特征数量: {len(feature_cols)}")
# print(f"数据形状: {df.shape}")
# print(f"目标变量分布: {df[target_variable].value_counts().to_dict()}")
#
# # 检查低方差特征
# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold(threshold=0.01)
# low_variance_features = []
# try:
#     selector.fit(df[feature_cols])
#     low_variance_mask = ~selector.get_support()
#     low_variance_features = [feature_cols[i] for i in range(len(feature_cols)) if low_variance_mask[i]]
# except:
#     pass
#
# def run_xgboost_with_optimization(df, target_variable, feature_cols):
#     """运行XGBoost模型训练和评估"""
#     print("\n=== XGBoost模型训练和评估 ===")
#
#     # 准备数据
#     X = df[feature_cols].copy()
#     y = df[target_variable].copy()
#
#     # 5折交叉验证
#     cv_scores = []
#     cv_f1_scores = []
#     cv_auc_scores = []
#     cv_recall_scores = []
#
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#         # 划分训练集和验证集
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#
#         # 标准化
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_val_scaled = scaler.transform(X_val)
#
#         # SMOTE处理类别不平衡
#         smote = SMOTE(random_state=42)
#         X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
#
#         # RFE特征选择
#         xgb_estimator = XGBClassifier(random_state=42, eval_metric='logloss')
#         rfe = RFE(estimator=xgb_estimator, n_features_to_select=min(30, len(feature_cols)), step=1)
#         X_train_rfe = rfe.fit_transform(X_train_balanced, y_train_balanced)
#         X_val_rfe = rfe.transform(X_val_scaled)
#
#         selected_features_rfe = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
#
#         # 训练XGBoost模型
#         xgb_model = XGBClassifier(
#             n_estimators=100,
#             max_depth=2,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42,
#             eval_metric='logloss'
#         )
#
#         xgb_model.fit(X_train_rfe, y_train_balanced)
#
#         # 预测
#         y_pred = xgb_model.predict(X_val_rfe)
#         y_pred_proba = xgb_model.predict_proba(X_val_rfe)[:, 1]
#
#         # 评估
#         accuracy = accuracy_score(y_val, y_pred)
#         f1 = f1_score(y_val, y_pred)
#         auc = roc_auc_score(y_val, y_pred_proba)
#         recall = recall_score(y_val, y_pred)
#
#         cv_scores.append(accuracy)
#         cv_f1_scores.append(f1)
#         cv_auc_scores.append(auc)
#         cv_recall_scores.append(recall)
#
#         print(f"[Fold {fold}] Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Recall={recall:.4f}")
#
#     # 输出交叉验证结果
#     print("\n=== 5-fold cross-validation results ===")
#     print(f"Acc: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
#     print(f"F1: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
#     print(f"AUC: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
#     print(f"Recall: {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}")
#
#     return {
#         'Accuracy': np.mean(cv_scores),
#         'F1': np.mean(cv_f1_scores),
#         'AUC': np.mean(cv_auc_scores),
#         'Recall': np.mean(cv_recall_scores)
#     }
#
# # 运行XGBoost模型
# results = run_xgboost_with_optimization(df, target_variable, feature_cols)
#
# # 添加特征重要性可视化
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置matplotlib后端
# import matplotlib
# matplotlib.use('Agg')
#
# # 训练最终模型获取特征重要性
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
#
# X = df[feature_cols]
# y = df[target_variable]
#
# # 数据预处理
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # SMOTE过采样
# smote = SMOTE(random_state=42, sampling_strategy=0.8)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
#
# # 训练XGBoost模型
# from xgboost import XGBClassifier
# xgb_model = XGBClassifier(
#     n_estimators=100,
#     max_depth=2,
#     learning_rate=0.03,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric='logloss'
# )
# xgb_model.fit(X_train_balanced, y_train_balanced)
#
# # 获取特征重要性
# feature_importance = xgb_model.feature_importances_
# feature_names = feature_cols
#
# # 创建特征重要性DataFrame
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': feature_importance
# }).sort_values('Importance', ascending=False)
#
# # 选择Top 20特征
# top20 = importance_df.head(20)
#
# # 绘制特征重要性图
# plt.figure(figsize=(10, 6))
# sns.barplot(x="Importance", y="Feature", hue="Feature", data=top20, palette="coolwarm", legend=False)
# plt.title("Top 20 Important Features (XGBoost)", fontsize=14)
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature Name")
# plt.tight_layout()
# plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# print("\n=== Top 20 Feature Importance (XGBoost) ===")
# for idx, row in top20.iterrows():
#     print(f"{row['Feature']}: {row['Importance']:.4f}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 1. 数据加载和预处理函数
# ----------------------------
def load_and_preprocess_data(file_path):
    """加载数据并进行预处理"""
    # 根据文件扩展名选择加载方式
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # 缺失值处理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # 数值型特征用中位数填充
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # 分类特征用众数填充
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')

    # 创建目标变量
    if 'buy' in df.columns:
        df['buy_target'] = (df['buy'] > 0).astype(int)
    elif 'buy_target' in df.columns:
        pass  # 已存在目标变量
    else:
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

# ----------------------------
# 数据加载和预处理
# ----------------------------
print("=== 数据加载和预处理 ===")
file_path = r"C:\Users\47556\Desktop\no_day1.xlsx"
df = load_and_preprocess_data(file_path)
target_variable = 'buy_target'

# ----------------------------
# 2. 特征工程和数据泄露防护
# ----------------------------
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
            df = df.drop(columns=[col])

    # 生成高阶交互特征
    # 选择数值型特征进行交互
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # 移除目标变量和排除列
    feature_cols = [col for col in numeric_features if col not in exclude_cols + leakage_cols]

    # 方差阈值过滤
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)

    # 先处理无穷大和NaN值
    df_features = df[feature_cols].copy()
    df_features = df_features.replace([np.inf, -np.inf], 0)
    df_features = df_features.fillna(0)

    # 应用方差过滤
    try:
        selector.fit(df_features)
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
    except:
        selected_features = feature_cols

    # 限制特征数量以避免过度复杂
    if len(selected_features) > 20:
        selected_features = selected_features[:20]

    # 生成2阶多项式特征
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    try:
        poly_features = poly.fit_transform(df_features[selected_features])
        poly_feature_names = poly.get_feature_names_out(selected_features)

        # 移除原始特征，只保留交互项
        interaction_names = [name for name in poly_feature_names if ' ' in name]
        interaction_features = poly_features[:, [i for i, name in enumerate(poly_feature_names) if ' ' in name]]

        # 添加交互特征到数据框
        for i, name in enumerate(interaction_names[:50]):  # 限制交互特征数量
            clean_name = name.replace(' ', '_x_')
            df[f'interaction_{clean_name}'] = interaction_features[:, i]

    except Exception as e:
        pass  # 静默处理多项式特征生成失败

    return df

# 应用特征工程
df = prepare_and_engineer_features(df)

# 最终特征选择
target_col = 'buy_target'
exclude_cols = [target_col, 'user_id']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 处理分类变量编码
le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

# 处理无穷大和NaN值
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# 异常值处理（IQR方法）
for col in feature_cols:
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 截断异常值而不是删除
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# 特征分布检查
print(f"\n=== 特征工程完成 ===")
print(f"最终特征数量: {len(feature_cols)}")
print(f"数据形状: {df.shape}")
print(f"目标变量分布: {df[target_variable].value_counts().to_dict()}")

# 检查低方差特征
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
low_variance_features = []
try:
    selector.fit(df[feature_cols])
    low_variance_mask = ~selector.get_support()
    low_variance_features = [feature_cols[i] for i in range(len(feature_cols)) if low_variance_mask[i]]
except:
    pass

def run_xgboost_with_optimization(df, target_variable, feature_cols):
    """运行XGBoost模型训练和评估"""
    print("\n=== XGBoost模型训练和评估 ===")

    # 准备数据
    X = df[feature_cols].copy()
    y = df[target_variable].copy()

    # 5折交叉验证
    cv_scores = []
    cv_f1_scores = []
    cv_auc_scores = []
    cv_recall_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # 划分训练集和验证集
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # SMOTE处理类别不平衡
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

        # RFE特征选择
        xgb_estimator = XGBClassifier(random_state=42, eval_metric='logloss')
        rfe = RFE(estimator=xgb_estimator, n_features_to_select=min(30, len(feature_cols)), step=1)
        X_train_rfe = rfe.fit_transform(X_train_balanced, y_train_balanced)
        X_val_rfe = rfe.transform(X_val_scaled)

        selected_features_rfe = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]

        # 训练XGBoost模型
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        xgb_model.fit(X_train_rfe, y_train_balanced)

        # 预测
        y_pred = xgb_model.predict(X_val_rfe)
        y_pred_proba = xgb_model.predict_proba(X_val_rfe)[:, 1]

        # 评估
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        recall = recall_score(y_val, y_pred)

        cv_scores.append(accuracy)
        cv_f1_scores.append(f1)
        cv_auc_scores.append(auc)
        cv_recall_scores.append(recall)

        print(f"[Fold {fold}] Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Recall={recall:.4f}")

    # 输出交叉验证结果
    print("\n=== 5-fold cross-validation results ===")
    print(f"Acc: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"F1: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
    print(f"AUC: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
    print(f"Recall: {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}")

    return {
        'Accuracy': np.mean(cv_scores),
        'F1': np.mean(cv_f1_scores),
        'AUC': np.mean(cv_auc_scores),
        'Recall': np.mean(cv_recall_scores)
    }

# 运行XGBoost模型
results = run_xgboost_with_optimization(df, target_variable, feature_cols)

# 添加特征重要性可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 训练最终模型获取特征重要性
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X = df[feature_cols]
y = df[target_variable]

# 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE过采样
smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# 训练XGBoost模型
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_balanced, y_train_balanced)

# 获取特征重要性
feature_importance = xgb_model.feature_importances_
feature_names = feature_cols

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# 选择Top 20特征
top20 = importance_df.head(20)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", hue="Feature", data=top20, palette="coolwarm", legend=False)
plt.title("Top 20 Important Features (XGBoost)", fontsize=14)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Top 20 Feature Importance (XGBoost) ===")
for idx, row in top20.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")
