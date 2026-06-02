import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ecommerce ML Dashboard", layout="wide")

# ═══════════════════════════════════════════════════════════════════
# i18n Dictionary
# ═══════════════════════════════════════════════════════════════════
I18N = {
    'zh': {
        'dashboard_title': '电商销售数据 — 机器学习仪表盘',
        'nav_overview': '数据总览',
        'nav_eda': '探索性分析',
        'nav_report': '分析报告',
        'nav_revenue': '收入预测（回归）',
        'nav_return': '退货预测（分类）',
        'nav_rating': '评分预测',
        'nav_predictor': '交互式预测',
        'nav_comparison': '模型对比',
        'data_info': '数据量',
        'data_orders': '条订单',
        'data_period': '2023–2025',
        'tech_stack': '技术栈',
        'models_used': '模型',
        'lang_switch': 'Language / 语言',
        # Data Overview
        'overview_title': '数据总览',
        'overview_subtitle': '合成电商销售数据集 (2023–2025)',
        'total_orders': '总订单数',
        'total_revenue': '总营收',
        'avg_order_value': '平均客单价',
        'return_rate': '退货率',
        'product_categories': '产品品类数',
        'regions': '地区数',
        'avg_rating': '平均评分',
        'avg_delivery_days': '平均配送天数',
        'raw_data_sample': '原始数据样本',
        'data_types_stats': '数据类型与统计',
        'column_types': '字段类型',
        'numerical_summary': '数值统计摘要',
        # EDA
        'eda_title': '探索性数据分析',
        'tab_revenue': '收入分析',
        'tab_product_region': '品类与地区',
        'tab_time': '时间趋势',
        'tab_correlation': '相关性',
        'revenue_distribution': '收入分布',
        'revenue_by_category': '按品类收入分布',
        'total_revenue_by_category': '各品类总收入',
        'total_revenue_by_region': '各地区总收入',
        'revenue_heatmap': '收入热力图：地区 × 品类',
        'return_rate_analysis': '退货率分析',
        'return_rate_by_category': '各品类退货率 (%)',
        'return_rate_by_region': '各地区退货率 (%)',
        'monthly_trends': '月度收入趋势',
        'revenue': '收入',
        'avg_order_value_chart': '平均客单价',
        'monthly_revenue_avg_order': '月度收入与平均客单价',
        'seasonal_patterns': '季节性规律',
        'avg_revenue_by_season': '各季节平均收入',
        'orders_by_season': '各季节订单数',
        'return_rate_by_season': '各季节退货率 (%)',
        'spring': '春季',
        'summer': '夏季',
        'fall': '秋季',
        'winter': '冬季',
        'correlation_matrix': '特征相关性矩阵',
        'key_relationships': '关键关系分析',
        'price_vs_revenue': '价格 vs 收入',
        'revenue_by_discount': '不同折扣等级的收入分布',
        # Revenue Prediction
        'revenue_title': '收入预测 — 随机森林回归',
        'revenue_desc': '基于产品、客户和交易特征预测订单收入',
        'model_performance': '模型性能',
        'r2_score': 'R² 得分',
        'mae': 'MAE',
        'rmse': 'RMSE',
        'mape': 'MAPE',
        'feature_importance_revenue': '特征重要性 — 收入预测',
        'feature_importance': '特征重要性（随机森林）',
        'interpretation': '解读',
        'revenue_interp': '''
        - **产品价格** 是收入预测的主导因素 — 价格越高，收入越高
        - **数量** 是第二重要特征 — 数量越多，收入越高
        - **折扣百分比** 对收入有负面影响（降价效应）
        - **产品品类**和**地区**影响适中
        - **时间特征**（月份、星期几）影响较小
        ''',
        'predicted_vs_actual': '预测值 vs 实际值',
        'actual_revenue': '实际收入',
        'predicted_revenue': '预测收入',
        'perfect_prediction': '完美预测',
        'residual_distribution': '残差分布',
        'residuals_vs_predicted': '残差 vs 预测值',
        # Return Prediction
        'return_title': '退货预测 — 随机森林分类器',
        'return_desc': '基于交易特征预测订单是否会被退货',
        'accuracy': '准确率',
        'precision': '精确率',
        'recall': '召回率',
        'f1_score': 'F1 得分',
        'roc_auc': 'ROC-AUC',
        'confusion_matrix': '混淆矩阵',
        'roc_curve': 'ROC 曲线',
        'predicted': '预测值',
        'actual': '实际值',
        'count': '数量',
        'not_returned': '未退货',
        'returned': '已退货',
        'true_negatives': '真阴性 (正确预测未退货)',
        'false_positives': '假阳性 (误报为退货)',
        'false_negatives': '假阴性 (漏报退货)',
        'true_positives': '真阳性 (正确预测退货)',
        'fpr': '假阳性率',
        'tpr': '真阳性率',
        'random_classifier': '随机分类器',
        'feature_importance_return': '特征重要性 — 退货预测',
        'return_interp': '''
        - **产品价格** 是退货的最强预测因素
        - **数量**和**配送天数**也有较大贡献
        - **产品品类**很重要 — 时尚品类退货率最高
        - **折扣百分比** 有一定的预测能力
        - 退货预测本质上比收入预测更难（正类仅占 6.06%，基线准确率约 94%）
        ''',
        'return_rate_by_key_features': '关键特征的退货率趋势',
        'return_rate_vs_discount': '退货率 vs 折扣%',
        'return_rate_vs_delivery': '退货率 vs 配送天数',
        'return_rate_vs_quantity': '退货率 vs 数量',
        'discount_pct': '折扣 %',
        'delivery_days': '配送天数',
        'quantity': '数量',
        'return_rate_pct': '退货率 (%)',
        # Rating Prediction
        'rating_title': '客户评分预测 — 随机森林回归',
        'rating_desc': '预测客户满意度评分 (2.0–5.0)',
        'rating_note': '''
        R² 得分接近零表明此合成数据集中的客户评分基本上是随机的
        — 无法从现有特征中可靠预测。这对于均匀分布的合成数据是预期的。
        在现实中，评分更多由产品质量、客服体验、开箱体验等因素驱动。
        ''',
        'rating_distribution': '评分分布',
        'avg_rating_by_category': '各品类平均评分',
        # Analysis Report
        'report_title': '分析报告',
        'report_subtitle': '基于当前筛选范围自动生成的经营摘要与行动建议',
        'report_filter_summary': '报告范围',
        'report_hero_label': '经营复盘',
        'report_filters': '报告筛选',
        'report_date_range': '时间范围',
        'report_region_filter': '地区',
        'report_category_filter': '品类',
        'report_payment_filter': '支付方式',
        'report_reset_filters': '重置筛选',
        'report_apply_filters': '应用筛选',
        'report_key_metrics': '核心指标',
        'report_key_insights': '核心结论',
        'report_health_diagnosis': '经营健康诊断',
        'report_topic_analysis': '专题分析',
        'report_actions': '行动建议',
        'report_total_quantity': '总销量',
        'report_empty_title': '当前筛选没有结果',
        'report_empty_desc': '请放宽时间、地区、品类或支付方式筛选条件后重试。',
        'report_low_sample_warning': '当前筛选样本较少，以下结论仅供参考。',
        'report_status_excellent': '优秀',
        'report_status_attention': '关注',
        'report_status_risk': '风险',
        'report_topic_category': '品类诊断',
        'report_topic_region': '地区诊断',
        'report_topic_season': '季节与月份诊断',
        'report_action_p0': 'P0 立即执行',
        'report_action_p1': 'P1 本月推进',
        'report_action_p2': 'P2 后续优化',
        'report_core_driver': '收入主力',
        'report_return_risk': '退货风险',
        'report_regional_signal': '区域信号',
        'report_time_opportunity': '时间机会',
        'report_discount_signal': '折扣表现',
        'report_single_dimension_note': '当前筛选仅包含单一维度，横向比较结论已弱化。',
        # Interactive Predictor
        'predictor_title': '交互式预测',
        'predictor_desc': '输入订单详情，获取实时机器学习预测',
        'order_details': '订单详情',
        'product_category': '产品品类',
        'region': '地区',
        'payment_method': '支付方式',
        'product_price': '产品价格',
        'discount_pct_label': '折扣百分比',
        'delivery_days_label': '配送天数',
        'order_month': '订单月份',
        'day_of_week': '星期几 (0=周一)',
        'predict_btn': '开始预测',
        'model_predictions': '模型预测结果',
        'predicted_revenue_label': '预测收入',
        'vs_base_price': 'vs 基础价格',
        'return_prediction': '退货预测',
        'likely_return': '高风险退货',
        'likely_keep': '低风险保留',
        'return_probability': '退货概率',
        'predicted_rating': '预测评分',
        'probability_breakdown': '退货概率分布',
        'input_summary': '输入参数摘要',
        'feature': '特征',
        'value': '值',
        'fill_instructions': '在左侧调整参数，点击预测按钮查看实时结果',
        'fill_revenue': '收入 — 该订单预计产生的收入',
        'fill_return': '退货风险 — 客户退货的概率',
        'fill_rating': '评分 — 预测的客户满意度评分',
        # Model Comparison
        'comparison_title': '模型对比与总结',
        'all_models': '三模型总览',
        'model': '模型',
        'type': '类型',
        'algorithm': '算法',
        'key_metric': '核心指标',
        'secondary_metric': '辅助指标',
        'revenue_model': '收入模型',
        'return_model': '退货模型',
        'top_features': '各模型 Top 特征',
        'key_conclusions': '核心结论',
        'conclusions_table': '''
        | 洞察 | 详情 |
        |------|------|
        | **收入高度可预测** | R² > 0.90 — 价格和数量解释大部分方差 |
        | **退货预测具有挑战性** | ROC-AUC ~0.58，合成数据中交易特征难以预测退货 |
        | **评分基本随机** | R² ≈ 0，评分在此数据中遵循均匀分布，与特征无关 |
        | **产品价格主导** | 对收入和退货预测都是最重要的特征 |
        | **合成数据局限性** | 品类/地区均匀分布限制了真实模式的发现 |
        ''',
        'sidebar_data': '数据',
        'sidebar_orders': '条订单',
    },
    'en': {
        'dashboard_title': 'Ecommerce Sales — ML Dashboard',
        'nav_overview': 'Data Overview',
        'nav_eda': 'Exploratory Analysis',
        'nav_report': 'Analysis Report',
        'nav_revenue': 'Revenue Prediction',
        'nav_return': 'Return Prediction',
        'nav_rating': 'Rating Prediction',
        'nav_predictor': 'Interactive Predictor',
        'nav_comparison': 'Model Comparison',
        'data_info': 'Data',
        'data_orders': 'orders',
        'data_period': '2023–2025',
        'tech_stack': 'Tech Stack',
        'models_used': 'Models',
        'lang_switch': 'Language / 语言',
        # Data Overview
        'overview_title': 'Data Overview',
        'overview_subtitle': 'Synthetic Ecommerce Sales Dataset (2023–2025)',
        'total_orders': 'Total Orders',
        'total_revenue': 'Total Revenue',
        'avg_order_value': 'Avg Order Value',
        'return_rate': 'Return Rate',
        'product_categories': 'Product Categories',
        'regions': 'Regions',
        'avg_rating': 'Avg Rating',
        'avg_delivery_days': 'Avg Delivery Days',
        'raw_data_sample': 'Raw Data Sample',
        'data_types_stats': 'Data Types & Statistics',
        'column_types': 'Column Types',
        'numerical_summary': 'Numerical Summary',
        # EDA
        'eda_title': 'Exploratory Data Analysis',
        'tab_revenue': 'Revenue Analysis',
        'tab_product_region': 'Product & Region',
        'tab_time': 'Time Trends',
        'tab_correlation': 'Correlation',
        'revenue_distribution': 'Revenue Distribution',
        'revenue_by_category': 'Revenue by Category',
        'total_revenue_by_category': 'Total Revenue by Category',
        'total_revenue_by_region': 'Total Revenue by Region',
        'revenue_heatmap': 'Revenue Heatmap: Region x Category',
        'return_rate_analysis': 'Return Rate Analysis',
        'return_rate_by_category': 'Return Rate by Category (%)',
        'return_rate_by_region': 'Return Rate by Region (%)',
        'monthly_trends': 'Monthly Revenue Trends',
        'revenue': 'Revenue',
        'avg_order_value_chart': 'Avg Order Value',
        'monthly_revenue_avg_order': 'Monthly Revenue & Avg Order Value',
        'seasonal_patterns': 'Seasonal Patterns',
        'avg_revenue_by_season': 'Avg Revenue by Season',
        'orders_by_season': 'Orders by Season',
        'return_rate_by_season': 'Return Rate by Season (%)',
        'spring': 'Spring',
        'summer': 'Summer',
        'fall': 'Fall',
        'winter': 'Winter',
        'correlation_matrix': 'Feature Correlation Matrix',
        'key_relationships': 'Key Relationships',
        'price_vs_revenue': 'Price vs Revenue',
        'revenue_by_discount': 'Revenue by Discount Level',
        # Revenue Prediction
        'revenue_title': 'Revenue Prediction — Random Forest Regressor',
        'revenue_desc': 'Predicting order revenue based on product, customer, and transaction features',
        'model_performance': 'Model Performance',
        'r2_score': 'R² Score',
        'mae': 'MAE',
        'rmse': 'RMSE',
        'mape': 'MAPE',
        'feature_importance_revenue': 'Feature Importance — Revenue Prediction',
        'feature_importance': 'Feature Importance (Random Forest)',
        'interpretation': 'Interpretation',
        'revenue_interp': '''
        - **Product Price** dominates revenue prediction — higher price drives higher revenue
        - **Quantity** is the second most important — more items means higher revenue
        - **Discount %** negatively impacts revenue (price reduction effect)
        - **Product Category** & **Region** have moderate influence
        - **Temporal features** (month, day of week) have minor impact
        ''',
        'predicted_vs_actual': 'Predicted vs Actual Revenue',
        'actual_revenue': 'Actual Revenue',
        'predicted_revenue': 'Predicted Revenue',
        'perfect_prediction': 'Perfect Prediction',
        'residual_distribution': 'Residual Distribution',
        'residuals_vs_predicted': 'Residuals vs Predicted Values',
        # Return Prediction
        'return_title': 'Return Prediction — Random Forest Classifier',
        'return_desc': 'Predicting whether an order will be returned based on transaction features',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'roc_auc': 'ROC-AUC',
        'confusion_matrix': 'Confusion Matrix',
        'roc_curve': 'ROC Curve',
        'predicted': 'Predicted',
        'actual': 'Actual',
        'count': 'Count',
        'not_returned': 'Not Returned',
        'returned': 'Returned',
        'true_negatives': 'True Negatives (correctly predicted not returned)',
        'false_positives': 'False Positives (predicted returned but was not)',
        'false_negatives': 'False Negatives (predicted not returned but was)',
        'true_positives': 'True Positives (correctly predicted returned)',
        'fpr': 'False Positive Rate',
        'tpr': 'True Positive Rate',
        'random_classifier': 'Random Classifier',
        'feature_importance_return': 'Feature Importance — Return Prediction',
        'return_interp': '''
        - **Product Price** is the strongest predictor of returns
        - **Quantity** and **Delivery Days** also contribute significantly
        - **Product Category** matters — Fashion has the highest return rate
        - **Discount %** has moderate predictive power
        - Return prediction is inherently harder than revenue prediction (6.06% positive class, ~94% baseline accuracy)
        ''',
        'return_rate_by_key_features': 'Return Rate by Key Features',
        'return_rate_vs_discount': 'Return Rate vs Discount %',
        'return_rate_vs_delivery': 'Return Rate vs Delivery Days',
        'return_rate_vs_quantity': 'Return Rate vs Quantity',
        'discount_pct': 'Discount %',
        'delivery_days': 'Delivery Days',
        'quantity': 'Quantity',
        'return_rate_pct': 'Return Rate (%)',
        # Rating Prediction
        'rating_title': 'Customer Rating Prediction — Random Forest Regressor',
        'rating_desc': 'Predicting customer satisfaction rating (2.0–5.0)',
        'rating_note': '''
        The R² score near zero indicates that customer ratings in this synthetic dataset
        are essentially random — they cannot be reliably predicted from the available features.
        This is expected for uniformly distributed synthetic data. In the real world, ratings
        are driven more by product quality, customer service, and unboxing experience rather
        than transactional attributes alone.
        ''',
        'rating_distribution': 'Rating Distribution',
        'avg_rating_by_category': 'Average Rating by Category',
        # Analysis Report
        'report_title': 'Analysis Report',
        'report_subtitle': 'Executive-style business summary and recommended actions for the current filters',
        'report_filter_summary': 'Report Scope',
        'report_hero_label': 'Executive Review',
        'report_filters': 'Report Filters',
        'report_date_range': 'Date Range',
        'report_region_filter': 'Region',
        'report_category_filter': 'Category',
        'report_payment_filter': 'Payment Method',
        'report_reset_filters': 'Reset Filters',
        'report_apply_filters': 'Apply Filters',
        'report_key_metrics': 'Key Metrics',
        'report_key_insights': 'Key Insights',
        'report_health_diagnosis': 'Business Health Diagnosis',
        'report_topic_analysis': 'Topic Analysis',
        'report_actions': 'Action Roadmap',
        'report_total_quantity': 'Total Units',
        'report_empty_title': 'No data for the current filters',
        'report_empty_desc': 'Broaden the date, region, category, or payment filters and try again.',
        'report_low_sample_warning': 'The current filter returns a small sample, so the insights below should be treated as directional only.',
        'report_status_excellent': 'Excellent',
        'report_status_attention': 'Attention',
        'report_status_risk': 'Risk',
        'report_topic_category': 'Category Diagnosis',
        'report_topic_region': 'Regional Diagnosis',
        'report_topic_season': 'Seasonality & Monthly Diagnosis',
        'report_action_p0': 'P0 Act Now',
        'report_action_p1': 'P1 This Month',
        'report_action_p2': 'P2 Next Step',
        'report_core_driver': 'Revenue Driver',
        'report_return_risk': 'Return Risk',
        'report_regional_signal': 'Regional Signal',
        'report_time_opportunity': 'Time Opportunity',
        'report_discount_signal': 'Discount Signal',
        'report_single_dimension_note': 'The current scope only includes one dimension, so comparison claims are intentionally softened.',
        # Interactive Predictor
        'predictor_title': 'Interactive Predictor',
        'predictor_desc': 'Input order details to get real-time ML predictions',
        'order_details': 'Order Details',
        'product_category': 'Product Category',
        'region': 'Region',
        'payment_method': 'Payment Method',
        'product_price': 'Product Price',
        'discount_pct_label': 'Discount %',
        'delivery_days_label': 'Delivery Days',
        'order_month': 'Order Month',
        'day_of_week': 'Day of Week (0=Mon)',
        'predict_btn': 'Predict',
        'model_predictions': 'Model Predictions',
        'predicted_revenue_label': 'Predicted Revenue',
        'vs_base_price': 'vs base price',
        'return_prediction': 'Return Prediction',
        'likely_return': 'Likely Return',
        'likely_keep': 'Likely Keep',
        'return_probability': 'Return Probability',
        'predicted_rating': 'Predicted Rating',
        'probability_breakdown': 'Return Probability Breakdown',
        'input_summary': 'Input Summary',
        'feature': 'Feature',
        'value': 'Value',
        'fill_instructions': 'Adjust the parameters on the left and click Predict to see real-time ML predictions for:',
        'fill_revenue': 'Revenue — how much this order will generate',
        'fill_return': 'Return Risk — probability of customer returning items',
        'fill_rating': 'Rating — predicted customer satisfaction score',
        # Model Comparison
        'comparison_title': 'Model Comparison & Summary',
        'all_models': 'All Models at a Glance',
        'model': 'Model',
        'type': 'Type',
        'algorithm': 'Algorithm',
        'key_metric': 'Key Metric',
        'secondary_metric': 'Secondary Metric',
        'revenue_model': 'Revenue Model',
        'return_model': 'Return Model',
        'top_features': 'Top Features Across Models',
        'key_conclusions': 'Key Conclusions',
        'conclusions_table': '''
        | Insight | Detail |
        |---------|--------|
        | **Revenue is highly predictable** | R² > 0.90 — price & quantity explain most variance |
        | **Return prediction is challenging** | ROC-AUC ~0.58 — returns are hard to predict from transactional data alone |
        | **Ratings are essentially random** | R² ≈ 0 — ratings follow a uniform distribution independent of features |
        | **Product Price dominates** | Most important feature for both revenue and return prediction |
        | **Synthetic data limitations** | Uniform distributions across categories/regions limit real-world pattern discovery |
        ''',
        'sidebar_data': 'Data',
        'sidebar_orders': 'orders',
    }
}

# ═══════════════════════════════════════════════════════════════════
# Language toggle in sidebar (BEFORE using any translated strings)
# ═══════════════════════════════════════════════════════════════════
if 'lang' not in st.session_state:
    st.session_state.lang = 'zh'

# Use a temporary sidebar just for the language switch at the very top
with st.sidebar:
    lang_prev = st.session_state.lang
    st.session_state.lang = 'zh' if st.toggle(
        'Language / 语言', value=True, help='中文 / English'
    ) else 'en'
    if st.session_state.lang != lang_prev:
        st.rerun()

T = I18N[st.session_state.lang]

# ═══════════════════════════════════════════════════════════════════
# Load & Cache Data
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_ecommerce_sales_2025.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['order_month'] = df['order_date'].dt.strftime('%Y-%m')
    df['quarter'] = df['order_date'].dt.quarter
    df['day_of_week'] = df['order_date'].dt.dayofweek
    season_key_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    df['season_key'] = df['month'].map(season_key_map)
    df['high_discount'] = (df['discount_percent'] >= 15).astype(int)
    df['price_per_unit'] = df['revenue'] / df['quantity']
    return df

df = load_data()

# ── Feature Engineering for ML ─────────────────────────────────────
def prepare_features(df):
    df_ml = df.copy()
    cat_cols = ['product_category', 'region', 'payment_method']
    le = {}
    for col in cat_cols:
        le[col] = LabelEncoder()
        df_ml[col + '_enc'] = le[col].fit_transform(df_ml[col])
    feature_cols = ['product_price', 'quantity', 'delivery_days', 'discount_percent',
                    'product_category_enc', 'region_enc', 'payment_method_enc',
                    'month', 'day_of_week']
    return df_ml, feature_cols, le

FEATURE_NAMES_EN = ['Product Price', 'Quantity', 'Delivery Days', 'Discount %',
                    'Product Category', 'Region', 'Payment Method', 'Month', 'Day of Week']
FEATURE_NAMES_ZH = ['产品价格', '数量', '配送天数', '折扣百分比',
                    '产品品类', '地区', '支付方式', '月份', '星期几']

# ── Train Models (cached) ──────────────────────────────────────────
@st.cache_resource
def train_models():
    X_reg = df_ml[feature_cols].values
    y_reg = df_ml['revenue'].values
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_r, y_train_r)
    y_pred_r = rf_reg.predict(X_test_r)

    reg_metrics = {
        'R² Score': round(r2_score(y_test_r, y_pred_r), 4),
        'MAE': round(mean_absolute_error(y_test_r, y_pred_r), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test_r, y_pred_r)), 2),
        'MAPE': round(np.mean(np.abs((y_test_r - y_pred_r) / y_test_r)) * 100, 2),
    }

    reg_importance = pd.DataFrame({
        'Feature': FEATURE_NAMES_EN,
        'Importance': rf_reg.feature_importances_
    }).sort_values('Importance', ascending=True)

    X_clf = df_ml[feature_cols].values
    y_clf = df_ml['is_returned'].values
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_c, y_train_c)
    y_pred_c = rf_clf.predict(X_test_c)
    y_prob_c = rf_clf.predict_proba(X_test_c)[:, 1]

    clf_metrics = {
        'Accuracy': round(accuracy_score(y_test_c, y_pred_c), 4),
        'Precision': round(precision_score(y_test_c, y_pred_c), 4),
        'Recall': round(recall_score(y_test_c, y_pred_c), 4),
        'F1 Score': round(f1_score(y_test_c, y_pred_c), 4),
        'ROC-AUC': round(roc_auc_score(y_test_c, y_prob_c), 4),
    }

    cm = confusion_matrix(y_test_c, y_pred_c)
    fpr, tpr, _ = roc_curve(y_test_c, y_prob_c)

    clf_importance = pd.DataFrame({
        'Feature': FEATURE_NAMES_EN,
        'Importance': rf_clf.feature_importances_
    }).sort_values('Importance', ascending=True)

    y_rating = df_ml['customer_rating'].values
    X_train_rt, X_test_rt, y_train_rt, y_test_rt = train_test_split(
        X_reg, y_rating, test_size=0.2, random_state=42)

    rf_rating = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_rating.fit(X_train_rt, y_train_rt)
    y_pred_rt = rf_rating.predict(X_test_rt)

    rating_metrics = {
        'R² Score': round(r2_score(y_test_rt, y_pred_rt), 4),
        'MAE': round(mean_absolute_error(y_test_rt, y_pred_rt), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test_rt, y_pred_rt)), 4),
    }

    return (rf_reg, reg_metrics, reg_importance,
            rf_clf, clf_metrics, clf_importance, cm, fpr, tpr,
            rf_rating, rating_metrics, label_encoders)

# ── Build localized feature importance dataframes ──────────────────
def localize_importance(importance_df):
    if st.session_state.lang == 'zh':
        mapping = dict(zip(FEATURE_NAMES_EN, FEATURE_NAMES_ZH))
        df_loc = importance_df.copy()
        df_loc['Feature'] = df_loc['Feature'].map(mapping)
        return df_loc
    return importance_df

# ── Cached EDA Computations ────────────────────────────────────────
@st.cache_data
def eda_revenue_by_category(_df):
    return _df.groupby('product_category')['revenue'].sum().sort_values(ascending=True)

@st.cache_data
def eda_revenue_by_region(_df):
    return _df.groupby('region')['revenue'].sum().sort_values(ascending=True)

@st.cache_data
def eda_pivot_region_category(_df):
    return _df.pivot_table(values='revenue', index='region', columns='product_category', aggfunc='sum')

@st.cache_data
def eda_return_by_category(_df):
    return (_df.groupby('product_category')['is_returned'].mean().sort_values(ascending=True) * 100).round(4)

@st.cache_data
def eda_return_by_region(_df):
    return (_df.groupby('region')['is_returned'].mean().sort_values(ascending=True) * 100).round(4)

@st.cache_data
def eda_monthly_trends(_df):
    monthly = _df.groupby(_df['order_date'].dt.to_period('M')).agg(
        revenue=('revenue', 'sum'),
        orders=('order_id', 'count'),
        avg_order=('revenue', 'mean')
    ).reset_index()
    monthly['order_date'] = monthly['order_date'].astype(str)
    return monthly

@st.cache_data
def eda_seasonal_avg_revenue(_df, season_order):
    return _df.groupby('season_key')['revenue'].mean().reindex(season_order)

@st.cache_data
def eda_seasonal_orders(_df, season_order):
    return _df.groupby('season_key')['order_id'].count().reindex(season_order)

@st.cache_data
def eda_seasonal_return(_df, season_order):
    return (_df.groupby('season_key')['is_returned'].mean().reindex(season_order) * 100).round(4)

@st.cache_data
def eda_time_trends(_df, season_order):
    monthly = _df.groupby('order_month', sort=True, observed=True).agg(
        revenue=('revenue', 'sum'),
        orders=('order_id', 'count'),
        avg_order=('revenue', 'mean')
    ).reset_index()
    monthly = monthly.rename(columns={'order_month': 'order_date'})
    seasonal = _df.groupby('season_key', sort=False, observed=True).agg(
        revenue=('revenue', 'mean'),
        orders=('order_id', 'count'),
        return_rate=('is_returned', 'mean')
    ).reindex(season_order)
    return monthly, seasonal['revenue'], seasonal['orders'], (seasonal['return_rate'] * 100).round(4)

@st.cache_data
def build_monthly_trends_figure(monthly, revenue_label, avg_order_label, chart_title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=monthly['order_date'], y=monthly['revenue'],
                             name=revenue_label, mode='lines',
                             line=dict(color='#636EFA', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly['order_date'], y=monthly['avg_order'],
                             name=avg_order_label, mode='lines',
                             line=dict(color='#EF553B', width=2, dash='dash')),
                  secondary_y=True)
    fig.update_layout(title=chart_title, height=450, hovermode='x unified')
    fig.update_xaxes(tickangle=45, dtick=2, type='category')
    fig.update_yaxes(title_text=f"{revenue_label} (USD)", secondary_y=False)
    fig.update_yaxes(title_text=f"{avg_order_label} (USD)", secondary_y=True)
    return fig

@st.cache_data
def build_season_bar_figure(x_values, y_values, title, color_scale, text_values):
    return px.bar(x=list(x_values), y=list(y_values),
                  title=title,
                  color=list(y_values), color_continuous_scale=color_scale,
                  text=list(text_values))

@st.cache_data
def eda_correlation_matrix(_df):
    numeric_cols = ['product_price', 'quantity', 'delivery_days', 'discount_percent',
                    'is_returned', 'customer_rating', 'revenue']
    return _df[numeric_cols].corr()

@st.cache_data
def eda_sample_for_scatter(_df, n=5000):
    return _df.sample(min(n, len(_df)), random_state=42)

@st.cache_data
def eda_revenue_sample(_df, n=5000):
    return _df.sample(min(n, len(_df)), random_state=42)

@st.cache_data
def eda_return_rate_by_feature(_df, col):
    return (_df.groupby(col)['is_returned'].mean() * 100).round(4)


def format_compact_number(value):
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def get_season_order_keys():
    return ('spring', 'summer', 'fall', 'winter')


def localize_season_labels(series_like):
    season_labels = {
        'spring': T['spring'],
        'summer': T['summer'],
        'fall': T['fall'],
        'winter': T['winter'],
    }
    localized = series_like.copy()
    localized.index = [season_labels.get(value, value) for value in localized.index]
    return localized


@st.cache_data
def compute_report_bundle(df):
    orders = len(df)
    revenue = float(df['revenue'].sum()) if orders else 0.0
    avg_order = float(df['revenue'].mean()) if orders else 0.0
    return_rate = float(df['is_returned'].mean() * 100) if orders else 0.0
    total_quantity = int(df['quantity'].sum()) if orders else 0
    avg_rating = float(df['customer_rating'].mean()) if orders else 0.0
    date_min = df['order_date'].min().date() if orders else None
    date_max = df['order_date'].max().date() if orders else None

    summary = {
        'orders': orders,
        'revenue': revenue,
        'avg_order_value': avg_order,
        'return_rate': return_rate,
        'total_quantity': total_quantity,
        'avg_rating': avg_rating,
        'is_empty': orders == 0,
        'is_low_sample': orders < 500,
        'date_min': date_min,
        'date_max': date_max,
    }

    bundle = {
        'summary': summary,
        'cat_revenue': pd.Series(dtype=float),
        'cat_return': pd.Series(dtype=float),
        'reg_revenue': pd.Series(dtype=float),
        'reg_return': pd.Series(dtype=float),
        'monthly': pd.DataFrame(columns=['order_date', 'revenue', 'orders', 'avg_order']),
        'seasonal_revenue': pd.Series(dtype=float),
        'seasonal_orders': pd.Series(dtype=float),
        'seasonal_return': pd.Series(dtype=float),
    }

    if not orders:
        return bundle

    cat_group = df.groupby('product_category').agg(
        revenue=('revenue', 'sum'),
        return_rate=('is_returned', 'mean')
    )
    reg_group = df.groupby('region').agg(
        revenue=('revenue', 'sum'),
        return_rate=('is_returned', 'mean')
    )
    monthly = df.groupby('order_month', sort=True, observed=True).agg(
        revenue=('revenue', 'sum'),
        orders=('order_id', 'count'),
        avg_order=('revenue', 'mean')
    ).reset_index()
    monthly = monthly.rename(columns={'order_month': 'order_date'})

    season_order = get_season_order_keys()
    season_group = df.groupby('season_key', sort=False, observed=True).agg(
        revenue=('revenue', 'mean'),
        orders=('order_id', 'count'),
        return_rate=('is_returned', 'mean')
    ).reindex(season_order)

    high_discount_df = df[df['discount_percent'] >= 15]
    high_discount_orders = len(high_discount_df)
    high_discount_revenue = float(high_discount_df['revenue'].sum()) if high_discount_orders else 0.0
    high_discount_return_rate = float(high_discount_df['is_returned'].mean() * 100) if high_discount_orders else 0.0

    top_category = cat_group['revenue'].idxmax()
    top_category_revenue = float(cat_group['revenue'].max())
    top_return_category = cat_group['return_rate'].idxmax()
    top_return_category_rate = float(cat_group['return_rate'].max() * 100)
    top_region = reg_group['revenue'].idxmax()
    top_region_revenue = float(reg_group['revenue'].max())
    top_return_region = reg_group['return_rate'].idxmax()
    top_return_region_rate = float(reg_group['return_rate'].max() * 100)
    top_month_row = monthly.loc[monthly['revenue'].idxmax()]
    top_month = str(top_month_row['order_date'])
    top_month_revenue = float(top_month_row['revenue'])

    summary.update({
        'top_category': top_category,
        'top_category_revenue': top_category_revenue,
        'top_category_share': top_category_revenue / revenue if revenue else 0.0,
        'top_return_category': top_return_category,
        'top_return_category_rate': top_return_category_rate,
        'top_region': top_region,
        'top_region_revenue': top_region_revenue,
        'top_return_region': top_return_region,
        'top_return_region_rate': top_return_region_rate,
        'top_month': top_month,
        'top_month_revenue': top_month_revenue,
        'high_discount_orders': high_discount_orders,
        'high_discount_revenue': high_discount_revenue,
        'high_discount_return_rate': high_discount_return_rate,
        'single_region': df['region'].nunique() == 1,
        'single_category': df['product_category'].nunique() == 1,
        'best_season_key': season_group['revenue'].idxmax(),
    })

    bundle.update({
        'summary': summary,
        'cat_revenue': cat_group['revenue'].sort_values(ascending=True),
        'cat_return': (cat_group['return_rate'].sort_values(ascending=True) * 100).round(4),
        'reg_revenue': reg_group['revenue'].sort_values(ascending=True),
        'reg_return': (reg_group['return_rate'].sort_values(ascending=True) * 100).round(4),
        'monthly': monthly,
        'seasonal_revenue': season_group['revenue'],
        'seasonal_orders': season_group['orders'],
        'seasonal_return': (season_group['return_rate'] * 100).round(4),
    })
    return bundle


def inject_report_styles():
    st.markdown("""
    <style>
    .report-hero,
    .report-card,
    .report-topic-card,
    .report-action-card,
    .report-empty {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(203, 213, 225, 0.86);
        border-radius: 22px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
        color: #0f172a;
    }
    .report-hero {
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
        padding: 1.6rem 1.75rem;
        background:
            radial-gradient(circle at 92% 8%, rgba(59, 130, 246, 0.16), transparent 34%),
            linear-gradient(135deg, #ffffff 0%, #f8fbff 48%, #eef6ff 100%);
        border-color: rgba(147, 197, 253, 0.7);
        box-shadow: 0 22px 55px rgba(37, 99, 235, 0.12);
    }
    .report-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background-image: linear-gradient(rgba(37, 99, 235, 0.06) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(37, 99, 235, 0.05) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(90deg, rgba(0,0,0,0.16), transparent 72%);
    }
    .report-hero > * {
        position: relative;
        z-index: 1;
    }
    .report-label {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        width: fit-content;
        font-size: 0.72rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #2563eb;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: 0.22rem 0.58rem;
        margin-bottom: 0.55rem;
        font-weight: 800;
    }
    .report-title {
        font-size: 2.12rem;
        font-weight: 850;
        letter-spacing: -0.04em;
        color: #0f172a;
        margin: 0 0 0.35rem 0;
    }
    .report-subtitle,
    .report-copy,
    .report-meta,
    .report-bullets li {
        color: #475569;
        line-height: 1.68;
    }
    .report-subtitle {
        max-width: 780px;
        font-size: 1.02rem;
        margin-bottom: 0.8rem;
    }
    .report-meta {
        display: inline-block;
        color: #334155;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(191, 219, 254, 0.82);
        border-radius: 12px;
        padding: 0.5rem 0.72rem;
        font-size: 0.9rem;
    }
    .report-metric {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid rgba(191, 219, 254, 0.9);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        min-height: 132px;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.07);
    }
    .report-metric-label {
        color: #64748b;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.42rem;
        font-weight: 800;
    }
    .report-metric-value {
        color: #1d4ed8;
        font-size: 1.78rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        margin-bottom: 0.45rem;
    }
    .report-metric-sub {
        color: #64748b;
        font-size: 0.88rem;
    }
    .report-card-title,
    .report-topic-title,
    .report-action-title {
        color: #0f172a;
        font-size: 1.08rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.45rem;
    }
    .report-kpi {
        color: #2563eb;
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.02em;
        margin-bottom: 0.45rem;
    }
    .report-status {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 800;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        margin-bottom: 0.7rem;
    }
    .report-status-excellent {
        color: #047857;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
    }
    .report-status-attention {
        color: #a16207;
        background: #fffbeb;
        border: 1px solid #fde68a;
    }
    .report-status-risk {
        color: #b91c1c;
        background: #fef2f2;
        border: 1px solid #fecaca;
    }
    .report-topic-kpi {
        color: #1d4ed8;
        font-size: 1.45rem;
        font-weight: 850;
        letter-spacing: -0.02em;
        margin: 0.25rem 0 0.65rem 0;
    }
    .report-action-priority {
        color: #2563eb;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 850;
        margin-bottom: 0.45rem;
    }
    .report-bullets {
        margin: 0.35rem 0 0 1rem;
        padding: 0;
    }
    .report-bullets li::marker {
        color: #60a5fa;
    }
    .report-empty {
        text-align: center;
        padding: 2rem 1.5rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    }
    .report-empty h3 {
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card_html(label, value, subtext):
    return f"""
    <div class=\"report-metric\">
        <div class=\"report-metric-label\">{label}</div>
        <div class=\"report-metric-value\">{value}</div>
        <div class=\"report-metric-sub\">{subtext}</div>
    </div>
    """


def report_card_html(title, kpi, body, accent=None):
    accent_html = f'<div class="report-label">{accent}</div>' if accent else ''
    return f"""
    <div class=\"report-card\">
        {accent_html}
        <div class=\"report-card-title\">{title}</div>
        <div class=\"report-kpi\">{kpi}</div>
        <div class=\"report-copy\">{body}</div>
    </div>
    """


def diagnosis_card_html(status, title, metric, body):
    status_class = {
        'excellent': 'report-status-excellent',
        'attention': 'report-status-attention',
        'risk': 'report-status-risk',
    }[status]
    status_label = {
        'excellent': T['report_status_excellent'],
        'attention': T['report_status_attention'],
        'risk': T['report_status_risk'],
    }[status]
    return f"""
    <div class=\"report-card\">
        <div class=\"report-status {status_class}\">{status_label}</div>
        <div class=\"report-card-title\">{title}</div>
        <div class=\"report-kpi\">{metric}</div>
        <div class=\"report-copy\">{body}</div>
    </div>
    """


def topic_card_html(label, title, kpi, body):
    return f"""
    <div class=\"report-topic-card\">
        <div class=\"report-label\">{label}</div>
        <div class=\"report-topic-title\">{title}</div>
        <div class=\"report-topic-kpi\">{kpi}</div>
        <div class=\"report-copy\">{body}</div>
    </div>
    """


def action_card_html(priority, title, bullets):
    bullet_html = ''.join(f'<li>{item}</li>' for item in bullets)
    return f"""
    <div class=\"report-action-card\">
        <div class=\"report-action-priority\">{priority}</div>
        <div class=\"report-action-title\">{title}</div>
        <ul class=\"report-bullets\">{bullet_html}</ul>
    </div>
    """


def render_report_filters(source_df):
    min_date = source_df['order_date'].min().date()
    max_date = source_df['order_date'].max().date()
    all_regions = sorted(source_df['region'].unique())
    all_categories = sorted(source_df['product_category'].unique())
    all_payments = sorted(source_df['payment_method'].unique())

    defaults = {
        'report_start_date': min_date,
        'report_end_date': max_date,
        'report_regions': all_regions,
        'report_categories': all_categories,
        'report_payments': all_payments,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.container(border=True):
        st.markdown(f"### {T['report_filters']}")
        with st.form('report_filters_form'):
            col1, col2 = st.columns(2)
            with col1:
                date_range = st.date_input(
                    T['report_date_range'],
                    value=(st.session_state.report_start_date, st.session_state.report_end_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = end_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range
            with col2:
                regions = st.multiselect(
                    T['report_region_filter'],
                    options=all_regions,
                    default=st.session_state.report_regions,
                )
                categories = st.multiselect(
                    T['report_category_filter'],
                    options=all_categories,
                    default=st.session_state.report_categories,
                )
                payments = st.multiselect(
                    T['report_payment_filter'],
                    options=all_payments,
                    default=st.session_state.report_payments,
                )

            apply_col, reset_col = st.columns(2)
            with apply_col:
                applied = st.form_submit_button(T['report_apply_filters'], use_container_width=True, type='primary')
            with reset_col:
                reset = st.form_submit_button(T['report_reset_filters'], use_container_width=True)

        if reset:
            st.session_state.report_start_date = min_date
            st.session_state.report_end_date = max_date
            st.session_state.report_regions = all_regions
            st.session_state.report_categories = all_categories
            st.session_state.report_payments = all_payments
        elif applied:
            st.session_state.report_start_date = start_date
            st.session_state.report_end_date = end_date
            st.session_state.report_regions = regions
            st.session_state.report_categories = categories
            st.session_state.report_payments = payments

    start_date = st.session_state.report_start_date
    end_date = st.session_state.report_end_date
    regions = st.session_state.report_regions
    categories = st.session_state.report_categories
    payments = st.session_state.report_payments

    filtered_df = source_df[
        (source_df['order_date'].dt.date >= start_date) &
        (source_df['order_date'].dt.date <= end_date) &
        (source_df['region'].isin(regions)) &
        (source_df['product_category'].isin(categories)) &
        (source_df['payment_method'].isin(payments))
    ].copy()

    filters_state = {
        'start_date': start_date,
        'end_date': end_date,
        'regions': regions,
        'categories': categories,
        'payments': payments,
    }
    return filtered_df, filters_state


def get_report_summary(filtered_df):
    return compute_report_bundle(filtered_df)['summary']


def build_report_insights(summary, filtered_df):
    if summary['is_empty']:
        return []

    insights = [
        {
            'title': T['report_core_driver'],
            'kpi': f"{summary['top_category']} · {summary['top_category_share'] * 100:.1f}%",
            'body': f"{summary['top_category']}贡献了当前筛选营收的{summary['top_category_share'] * 100:.1f}%，营收规模约{format_compact_number(summary['top_category_revenue'])}。" if st.session_state.lang == 'zh' else f"{summary['top_category']} contributes {summary['top_category_share'] * 100:.1f}% of revenue in the current scope, or about {format_compact_number(summary['top_category_revenue'])}.",
            'accent': 'Executive Summary'
        },
        {
            'title': T['report_return_risk'],
            'kpi': f"{summary['top_return_category_rate']:.2f}%",
            'body': f"{summary['top_return_category']}是当前退货率最高的品类，建议优先检查商品描述、尺码或履约体验。" if st.session_state.lang == 'zh' else f"{summary['top_return_category']} has the highest return rate in the current slice, so product detail, fit, or fulfillment quality should be checked first.",
            'accent': 'Risk Signal'
        },
        {
            'title': T['report_regional_signal'],
            'kpi': f"{summary['top_region']} / {summary['top_return_region']}",
            'body': f"{summary['top_region']}贡献最高营收，而{summary['top_return_region']}退货率最高，为{summary['top_return_region_rate']:.2f}%。" if st.session_state.lang == 'zh' else f"{summary['top_region']} leads revenue while {summary['top_return_region']} has the highest return rate at {summary['top_return_region_rate']:.2f}%.",
            'accent': 'Regional'
        },
        {
            'title': T['report_time_opportunity'],
            'kpi': summary['top_month'],
            'body': f"{summary['top_month']}是当前筛选下表现最强的月份，营收约{format_compact_number(summary['top_month_revenue'])}。" if st.session_state.lang == 'zh' else f"{summary['top_month']} is the strongest month in the current scope, generating about {format_compact_number(summary['top_month_revenue'])} in revenue.",
            'accent': 'Timing'
        },
    ]

    if summary['high_discount_orders'] > 0:
        insights.append({
            'title': T['report_discount_signal'],
            'kpi': f"{summary['high_discount_return_rate']:.2f}%",
            'body': f"15%及以上折扣订单共有{summary['high_discount_orders']:,}笔，退货率为{summary['high_discount_return_rate']:.2f}%。" if st.session_state.lang == 'zh' else f"Orders with discounts of 15% or more total {summary['high_discount_orders']:,}, with a return rate of {summary['high_discount_return_rate']:.2f}%.",
            'accent': 'Discount'
        })

    return insights


def build_report_diagnosis(summary, report_bundle):
    if summary['is_empty']:
        return []

    diagnosis = []

    revenue_share = summary['top_category_share'] * 100
    if revenue_share < 20:
        revenue_status = 'excellent'
    elif revenue_share < 30:
        revenue_status = 'attention'
    else:
        revenue_status = 'risk'
    diagnosis.append({
        'status': revenue_status,
        'title': T['report_core_driver'],
        'metric': f"{revenue_share:.1f}%",
        'body': f"头部品类营收占比为{revenue_share:.1f}%，用于判断业务是否过度集中。" if st.session_state.lang == 'zh' else f"The top category contributes {revenue_share:.1f}% of revenue, which helps judge whether the business mix is overly concentrated.",
    })

    if summary['return_rate'] < 5.5:
        return_status = 'excellent'
    elif summary['return_rate'] < 8:
        return_status = 'attention'
    else:
        return_status = 'risk'
    diagnosis.append({
        'status': return_status,
        'title': T['report_return_risk'],
        'metric': f"{summary['return_rate']:.2f}%",
        'body': f"整体退货率为{summary['return_rate']:.2f}%，高风险品类为{summary['top_return_category']}。" if st.session_state.lang == 'zh' else f"The overall return rate is {summary['return_rate']:.2f}%, with {summary['top_return_category']} acting as the highest-risk category.",
    })

    seasonal_revenue = report_bundle['seasonal_revenue']
    seasonal_orders = report_bundle['seasonal_orders']
    best_season_key = summary['best_season_key']
    best_season = {'spring': T['spring'], 'summer': T['summer'], 'fall': T['fall'], 'winter': T['winter']}[best_season_key]
    season_status = 'excellent' if seasonal_orders.max() >= seasonal_orders.mean() else 'attention'
    diagnosis.append({
        'status': season_status,
        'title': T['report_time_opportunity'],
        'metric': str(best_season),
        'body': f"{best_season}同时承接了更高的营收表现，可作为备货和营销的优先窗口。" if st.session_state.lang == 'zh' else f"{best_season} carries the strongest revenue signal and can be prioritized for inventory and campaign planning.",
    })

    region_status = 'attention' if summary['single_region'] else ('excellent' if summary['top_return_region_rate'] < 6.5 else 'risk')
    diagnosis.append({
        'status': region_status,
        'title': T['report_regional_signal'],
        'metric': f"{summary['top_return_region_rate']:.2f}%",
        'body': T['report_single_dimension_note'] if summary['single_region'] else (f"{summary['top_return_region']}是退货率最高地区，需要检查履约或品类结构差异。" if st.session_state.lang == 'zh' else f"{summary['top_return_region']} has the highest regional return rate, so fulfillment quality or category mix should be reviewed."),
    })

    rating_status = 'excellent' if summary['avg_rating'] >= 3.5 else 'attention'
    diagnosis.append({
        'status': rating_status,
        'title': T['avg_rating'],
        'metric': f"{summary['avg_rating']:.2f}",
        'body': f"平均评分为{summary['avg_rating']:.2f}，适合作为客户体验的辅助信号，而非单独决策依据。" if st.session_state.lang == 'zh' else f"The average rating is {summary['avg_rating']:.2f}, which is useful as a support signal for customer experience rather than a standalone decision metric.",
    })

    return diagnosis


def build_report_actions(summary):
    if summary['is_empty']:
        return []

    p0_title = f"聚焦 {summary['top_return_category']} 退货治理" if st.session_state.lang == 'zh' else f"Focus on {summary['top_return_category']} return reduction"
    p1_title = f"放大 {summary['top_month']} 的销售打法" if st.session_state.lang == 'zh' else f"Scale the {summary['top_month']} playbook"
    p2_title = '建立持续监控节奏' if st.session_state.lang == 'zh' else 'Establish an operating rhythm'

    actions = [
        {
            'priority': T['report_action_p0'],
            'title': p0_title,
            'bullets': [
                f"复核 {summary['top_return_category']} 的商品描述、尺码或售后原因。" if st.session_state.lang == 'zh' else f"Audit product detail, sizing, and after-sales reasons for {summary['top_return_category']}.",
                f"优先检查 {summary['top_return_region']} 区域内的高退货订单样本。" if st.session_state.lang == 'zh' else f"Review high-return orders from {summary['top_return_region']} first.",
                T['report_low_sample_warning'] if summary['is_low_sample'] else ("将退货问题按履约、商品、用户预期三类做归因。" if st.session_state.lang == 'zh' else "Classify return causes into fulfillment, product, and expectation gaps."),
            ]
        },
        {
            'priority': T['report_action_p1'],
            'title': p1_title,
            'bullets': [
                f"围绕 {summary['top_month']} 复盘库存、折扣和营销配合。" if st.session_state.lang == 'zh' else f"Review inventory, discounting, and campaign coordination around {summary['top_month']}.",
                f"优先复制 {summary['top_category']} 的主力打法到相邻品类。" if st.session_state.lang == 'zh' else f"Replicate what works in {summary['top_category']} into adjacent categories.",
                f"结合 {summary['top_region']} 的表现评估区域资源分配。" if st.session_state.lang == 'zh' else f"Use {summary['top_region']} performance to rebalance regional investment.",
            ]
        },
        {
            'priority': T['report_action_p2'],
            'title': p2_title,
            'bullets': [
                "固定每周刷新一次报告页，持续观察核心指标波动。" if st.session_state.lang == 'zh' else "Refresh the report weekly to monitor movement in the core metrics.",
                "把高折扣订单和退货率放到同一监控视角下观察。" if st.session_state.lang == 'zh' else "Track high-discount orders and return rate in the same operating view.",
                "当筛选范围收窄时，优先把结论作为方向性信号而不是最终判断。" if st.session_state.lang == 'zh' else "When the scope becomes narrow, treat the conclusions as directional rather than final.",
            ]
        },
    ]
    return actions


def render_report_page(source_df):
    inject_report_styles()
    filtered_df, filters_state = render_report_filters(source_df)
    report_bundle = compute_report_bundle(filtered_df)
    summary = report_bundle['summary']

    if summary['is_empty']:
        st.markdown(f"""
        <div class=\"report-empty\">
            <div class=\"report-label\">Analysis Report</div>
            <h3>{T['report_empty_title']}</h3>
            <p class=\"report-copy\">{T['report_empty_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        return

    filter_summary = f"{filters_state['start_date']} → {filters_state['end_date']} · {len(filters_state['regions'])} {T['regions']} · {len(filters_state['categories'])} {T['product_categories']}"
    revenue_subtext = (f"营收规模 {format_compact_number(summary['revenue'])}" if st.session_state.lang == 'zh' else f"Gross revenue {format_compact_number(summary['revenue'])}")
    st.markdown(f"""
    <div class=\"report-hero\">
        <div class=\"report-label\">{T['report_hero_label']}</div>
        <div class=\"report-title\">{T['report_title']}</div>
        <div class=\"report-subtitle\">{T['report_subtitle']}</div>
        <div class=\"report-meta\"><strong>{T['report_filter_summary']}:</strong> {filter_summary}</div>
    </div>
    """, unsafe_allow_html=True)

    if summary['is_low_sample']:
        st.warning(T['report_low_sample_warning'])

    st.markdown(f"### {T['report_key_metrics']}")
    metric_specs = [
        (T['total_orders'], f"{summary['orders']:,}", f"{summary['date_min']} → {summary['date_max']}"),
        (T['total_revenue'], f"¥{summary['revenue']:,.0f}", revenue_subtext),
        (T['avg_order_value'], f"¥{summary['avg_order_value']:,.2f}", T['report_core_driver']),
        (T['return_rate'], f"{summary['return_rate']:.2f}%", summary['top_return_category']),
        (T['report_total_quantity'], f"{summary['total_quantity']:,}", T['quantity']),
        (T['avg_rating'], f"{summary['avg_rating']:.2f}", T['report_regional_signal']),
    ]
    metric_cols = st.columns(6)
    for col, (label, value, subtext) in zip(metric_cols, metric_specs):
        with col:
            st.markdown(metric_card_html(label, value, subtext), unsafe_allow_html=True)

    st.markdown(f"### {T['report_key_insights']}")
    insights = build_report_insights(summary, filtered_df)
    insight_cols = st.columns(2)
    for idx, insight in enumerate(insights):
        with insight_cols[idx % 2]:
            st.markdown(report_card_html(insight['title'], insight['kpi'], insight['body'], insight.get('accent')), unsafe_allow_html=True)

    st.markdown(f"### {T['report_health_diagnosis']}")
    diagnosis = build_report_diagnosis(summary, report_bundle)
    diagnosis_cols = st.columns(3)
    for idx, item in enumerate(diagnosis):
        with diagnosis_cols[idx % 3]:
            st.markdown(diagnosis_card_html(item['status'], item['title'], item['metric'], item['body']), unsafe_allow_html=True)

    st.markdown(f"### {T['report_topic_analysis']}")
    topic_col1, topic_col2, topic_col3 = st.columns(3)
    with topic_col1:
        st.markdown(topic_card_html(
            'Category',
            T['report_topic_category'],
            f"{summary['top_category']} · {summary['top_category_share'] * 100:.1f}%",
            (f"当前收入主力是{summary['top_category']}，而{summary['top_return_category']}退货率最高，说明增长与风险并不完全重叠。" if st.session_state.lang == 'zh' else f"{summary['top_category']} is the core revenue driver while {summary['top_return_category']} carries the highest return rate, so growth and risk do not fully overlap.") if not summary['single_category'] else T['report_single_dimension_note']
        ), unsafe_allow_html=True)
        cat_rev = report_bundle['cat_revenue']
        fig = px.bar(x=cat_rev.values, y=cat_rev.index, orientation='h', color=cat_rev.values,
                     color_continuous_scale='Blues')
        fig.update_layout(
            height=280,
            template='plotly_white',
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(255,255,255,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            font=dict(color='#334155'),
        )
        fig.update_xaxes(gridcolor='rgba(148,163,184,0.18)', zerolinecolor='rgba(148,163,184,0.25)')
        fig.update_yaxes(gridcolor='rgba(148,163,184,0.12)')
        st.plotly_chart(fig, use_container_width=True)
    with topic_col2:
        st.markdown(topic_card_html(
            'Region',
            T['report_topic_region'],
            f"{summary['top_region']} / {summary['top_return_region_rate']:.2f}%",
            T['report_single_dimension_note'] if summary['single_region'] else (f"{summary['top_region']}承接最高营收，{summary['top_return_region']}则是退货风险最高地区。" if st.session_state.lang == 'zh' else f"{summary['top_region']} leads revenue while {summary['top_return_region']} has the highest return risk.")
        ), unsafe_allow_html=True)
        reg_rev = report_bundle['reg_revenue']
        fig = px.bar(x=reg_rev.values, y=reg_rev.index, orientation='h', color=reg_rev.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(
            height=280,
            template='plotly_white',
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(255,255,255,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            font=dict(color='#334155'),
        )
        fig.update_xaxes(gridcolor='rgba(148,163,184,0.18)', zerolinecolor='rgba(148,163,184,0.25)')
        fig.update_yaxes(gridcolor='rgba(148,163,184,0.12)')
        st.plotly_chart(fig, use_container_width=True)
    with topic_col3:
        st.markdown(topic_card_html(
            'Seasonality',
            T['report_topic_season'],
            summary['top_month'],
            f"{summary['top_month']}营收约为{format_compact_number(summary['top_month_revenue'])}，可作为阶段性投放与备货锚点。" if st.session_state.lang == 'zh' else f"{summary['top_month']} delivers about {format_compact_number(summary['top_month_revenue'])} in revenue and can anchor campaign and inventory timing."
        ), unsafe_allow_html=True)
        monthly = report_bundle['monthly']
        fig = px.line(monthly, x='order_date', y='revenue', markers=True)
        fig.update_layout(
            height=280,
            template='plotly_white',
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(255,255,255,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title='',
            yaxis_title='',
            font=dict(color='#334155'),
        )
        fig.update_xaxes(
            tickangle=45,
            dtick=2,
            type='category',
            gridcolor='rgba(148,163,184,0.18)',
            zerolinecolor='rgba(148,163,184,0.25)',
        )
        fig.update_yaxes(gridcolor='rgba(148,163,184,0.18)', zerolinecolor='rgba(148,163,184,0.25)')
        fig.update_traces(line=dict(color='#2563eb', width=3), marker=dict(size=7, color='#2563eb'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### {T['report_actions']}")
    action_cols = st.columns(3)
    actions = build_report_actions(summary)
    for col, action in zip(action_cols, actions):
        with col:
            st.markdown(action_card_html(action['priority'], action['title'], action['bullets']), unsafe_allow_html=True)

# ── Sidebar Navigation ─────────────────────────────────────────────
st.sidebar.title(T['dashboard_title'])
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    T['nav_overview'],
    T['nav_eda'],
    T['nav_report'],
    T['nav_revenue'],
    T['nav_return'],
    T['nav_rating'],
    T['nav_predictor'],
    T['nav_comparison'],
], label_visibility="collapsed")

# Build nav labels for comparison
NAV_KEYS = ['nav_overview', 'nav_eda', 'nav_report', 'nav_revenue', 'nav_return',
            'nav_rating', 'nav_predictor', 'nav_comparison']
NAV_LABELS = {T[k]: k for k in NAV_KEYS}
PAGE_KEY = NAV_LABELS.get(page, 'nav_overview')

MODEL_PAGES = {'nav_revenue', 'nav_return', 'nav_rating', 'nav_predictor', 'nav_comparison'}
if PAGE_KEY in MODEL_PAGES:
    df_ml, feature_cols, label_encoders = prepare_features(df)
    (rf_reg, reg_metrics, reg_importance,
     rf_clf, clf_metrics, clf_importance, cm, fpr, tpr,
     rf_rating, rating_metrics, label_encoders) = train_models()
    reg_importance_loc = localize_importance(reg_importance)
    clf_importance_loc = localize_importance(clf_importance)

st.sidebar.markdown("---")
st.sidebar.caption(f"{T['sidebar_data']}: {len(df):,} {T['sidebar_orders']} | {T['data_period']}")

# ═══════════════════════════════════════════════════════════════════
# PAGE 1: Data Overview
# ═══════════════════════════════════════════════════════════════════
if PAGE_KEY == 'nav_overview':
    st.title(T['overview_title'])
    st.markdown(f"### {T['overview_subtitle']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(T['total_orders'], f"{len(df):,}")
    with col2:
        st.metric(T['total_revenue'], f"{df['revenue'].sum():,.0f}")
    with col3:
        st.metric(T['avg_order_value'], f"{df['revenue'].mean():,.2f}")
    with col4:
        st.metric(T['return_rate'], f"{df['is_returned'].mean()*100:.2f}%")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(T['product_categories'], df['product_category'].nunique())
    with col2:
        st.metric(T['regions'], df['region'].nunique())
    with col3:
        st.metric(T['avg_rating'], f"{df['customer_rating'].mean():.2f}")
    with col4:
        st.metric(T['avg_delivery_days'], f"{df['delivery_days'].mean():.1f}")

    st.markdown("---")
    st.markdown(f"### {T['raw_data_sample']}")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown(f"### {T['data_types_stats']}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{T['column_types']}:**")
        st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']).reset_index().rename(columns={'index': 'Column'}),
                     use_container_width=True)
    with col2:
        st.write(f"**{T['numerical_summary']}:**")
        st.dataframe(df.describe(), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 2: Exploratory Analysis
# ═══════════════════════════════════════════════════════════════════
elif PAGE_KEY == 'nav_eda':
    st.title(T['eda_title'])

    eda_tab = st.radio(
        "",
        [T['tab_revenue'], T['tab_product_region'], T['tab_time'], T['tab_correlation']],
        horizontal=True,
        label_visibility="collapsed",
    )

    if eda_tab == T['tab_revenue']:
        cat_rev = eda_revenue_by_category(df)
        reg_rev = eda_revenue_by_region(df)
        revenue_sample = eda_revenue_sample(df)

        st.markdown(f"### {T['revenue_distribution']}")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='revenue', nbins=50,
                               title=T['revenue_distribution'],
                               color_discrete_sequence=['#636EFA'], marginal='box')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(revenue_sample, y='revenue', x='product_category',
                         title=T['revenue_by_category'],
                         color='product_category', height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {T['total_revenue_by_category']}")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=cat_rev.values, y=cat_rev.index, orientation='h',
                         title=T['total_revenue_by_category'],
                         color=cat_rev.values, color_continuous_scale='Blues',
                         text_auto='.2s')
            fig.update_layout(height=380, xaxis_title=T['revenue'], yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=reg_rev.values, y=reg_rev.index, orientation='h',
                         title=T['total_revenue_by_region'],
                         color=reg_rev.values, color_continuous_scale='Greens',
                         text_auto='.2s')
            fig.update_layout(height=380, xaxis_title=T['revenue'], yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == T['tab_product_region']:
        pivot_data = eda_pivot_region_category(df)
        ret_cat = eda_return_by_category(df)
        ret_reg = eda_return_by_region(df)

        st.markdown(f"### {T['revenue_heatmap']}")
        fig = px.imshow(pivot_data, aspect='auto',
                        title=T['revenue_heatmap'],
                        color_continuous_scale='RdBu_r')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {T['return_rate_analysis']}")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=ret_cat.values, y=ret_cat.index, orientation='h',
                         title=T['return_rate_by_category'],
                         color=ret_cat.values, color_continuous_scale='Reds',
                         text=ret_cat.values.round(2))
            fig.update_layout(height=380, xaxis_title=T['return_rate_pct'], yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=ret_reg.values, y=ret_reg.index, orientation='h',
                         title=T['return_rate_by_region'],
                         color=ret_reg.values, color_continuous_scale='Oranges',
                         text=ret_reg.values.round(2))
            fig.update_layout(height=380, xaxis_title=T['return_rate_pct'], yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == T['tab_time']:
        season_order = get_season_order_keys()
        monthly, sz_rev, sz_ord, sz_ret = eda_time_trends(df, season_order)
        sz_rev = localize_season_labels(sz_rev)
        sz_ord = localize_season_labels(sz_ord)
        sz_ret = localize_season_labels(sz_ret)

        st.markdown(f"### {T['monthly_trends']}")
        fig = build_monthly_trends_figure(
            monthly,
            T['revenue'],
            T['avg_order_value_chart'],
            T['monthly_revenue_avg_order'],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {T['seasonal_patterns']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = build_season_bar_figure(
                tuple(sz_rev.index), tuple(sz_rev.values),
                T['avg_revenue_by_season'],
                'Viridis',
                tuple(sz_rev.values.round(0)),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = build_season_bar_figure(
                tuple(sz_ord.index), tuple(sz_ord.values),
                T['orders_by_season'],
                'Viridis',
                tuple(sz_ord.values),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = build_season_bar_figure(
                tuple(sz_ret.index), tuple(sz_ret.values),
                T['return_rate_by_season'],
                'Reds',
                tuple(sz_ret.values.round(2)),
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        corr = eda_correlation_matrix(df)
        sample_df = eda_sample_for_scatter(df)
        revenue_sample = eda_revenue_sample(df)

        st.markdown(f"### {T['correlation_matrix']}")
        fig = px.imshow(corr, aspect='auto',
                        title=T['correlation_matrix'],
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### {T['key_relationships']}")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(sample_df, x='product_price', y='revenue',
                             color='product_category',
                             title=T['price_vs_revenue'], opacity=0.6, size='quantity',
                             hover_data=['region'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(revenue_sample, x='discount_percent', y='revenue', color='discount_percent',
                         title=T['revenue_by_discount'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 3: Analysis Report
# ═══════════════════════════════════════════════════════════════════
elif PAGE_KEY == 'nav_report':
    render_report_page(df)

# ═══════════════════════════════════════════════════════════════════
# PAGE 4: Revenue Prediction
# ═══════════════════════════════════════════════════════════════════
elif PAGE_KEY == 'nav_revenue':
    st.title(T['revenue_title'])
    st.markdown(f"*{T['revenue_desc']}*")

    st.markdown(f"### {T['model_performance']}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(T['r2_score'], reg_metrics['R² Score'])
    with col2:
        st.metric(f"{T['mae']} (USD)", reg_metrics['MAE'])
    with col3:
        st.metric(f"{T['rmse']} (USD)", reg_metrics['RMSE'])
    with col4:
        st.metric(f"{T['mape']} (%)", f"{reg_metrics['MAPE']}%")

    st.markdown("---")
    st.markdown(f"### {T['feature_importance_revenue']}")
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.bar(reg_importance_loc, x='Importance', y='Feature', orientation='h',
                     title=T['feature_importance'],
                     color='Importance', color_continuous_scale='Blues',
                     text=reg_importance_loc['Importance'].round(4))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"#### {T['interpretation']}")
        st.markdown(T['revenue_interp'])

    st.markdown(f"### {T['predicted_vs_actual']}")
    X_test_r = df_ml[feature_cols].values
    y_test_r = df_ml['revenue'].values
    _, X_test_r_sub, _, y_test_r_sub = train_test_split(
        X_test_r, y_test_r, test_size=0.2, random_state=42)
    y_pred_sample = rf_reg.predict(X_test_r_sub)

    fig = px.scatter(x=y_test_r_sub, y=y_pred_sample, opacity=0.4,
                     title=T['predicted_vs_actual'],
                     labels={'x': T['actual_revenue'] + ' (USD)',
                             'y': T['predicted_revenue'] + ' (USD)'},
                     color=np.abs(y_test_r_sub - y_pred_sample),
                     color_continuous_scale='Reds')
    fig.add_trace(go.Scatter(x=[0, 3000], y=[0, 3000], mode='lines',
                             name=T['perfect_prediction'],
                             line=dict(dash='dash', color='green')))
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### {T['residual_distribution']}")
    residuals = y_test_r_sub - y_pred_sample
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(residuals, nbins=80, title=T['residual_distribution'],
                           color_discrete_sequence=['#636EFA'])
        fig.add_vline(x=0, line_dash='dash', line_color='red')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(x=y_pred_sample, y=residuals, opacity=0.4,
                         title=T['residuals_vs_predicted'],
                         labels={'x': T['predicted_revenue'] + ' (USD)',
                                 'y': 'Residual (USD)'})
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 4: Return Prediction
# ═══════════════════════════════════════════════════════════════════
elif PAGE_KEY == 'nav_return':
    st.title(T['return_title'])
    st.markdown(f"*{T['return_desc']}*")

    st.markdown(f"### {T['model_performance']}")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(T['accuracy'], clf_metrics['Accuracy'])
    with col2:
        st.metric(T['precision'], clf_metrics['Precision'])
    with col3:
        st.metric(T['recall'], clf_metrics['Recall'])
    with col4:
        st.metric(T['f1_score'], clf_metrics['F1 Score'])
    with col5:
        st.metric(T['roc_auc'], clf_metrics['ROC-AUC'])

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {T['confusion_matrix']}")
        fig = px.imshow(cm, text_auto=True, aspect='auto',
                        title=T['confusion_matrix'],
                        labels=dict(x=T['predicted'], y=T['actual'], color=T['count']),
                        x=[T['not_returned'], T['returned']],
                        y=[T['not_returned'], T['returned']],
                        color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        - **{T['true_negatives']}**: {cm[0][0]:,}
        - **{T['false_positives']}**: {cm[0][1]:,}
        - **{T['false_negatives']}**: {cm[1][0]:,}
        - **{T['true_positives']}**: {cm[1][1]:,}
        """)

    with col2:
        st.markdown(f"### {T['roc_curve']}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"ROC (AUC={clf_metrics['ROC-AUC']})",
                                 line=dict(color='#636EFA', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name=T['random_classifier'],
                                 line=dict(dash='dash', color='gray')))
        fig.update_layout(title=T['roc_curve'],
                          xaxis_title=T['fpr'], yaxis_title=T['tpr'],
                          height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### {T['feature_importance_return']}")
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = px.bar(clf_importance_loc, x='Importance', y='Feature', orientation='h',
                     title=T['feature_importance'],
                     color='Importance', color_continuous_scale='Oranges',
                     text=clf_importance_loc['Importance'].round(4))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"#### {T['interpretation']}")
        st.markdown(T['return_interp'])

    st.markdown(f"### {T['return_rate_by_key_features']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        ret_disc = df.groupby('discount_percent')['is_returned'].mean() * 100
        fig = px.line(x=ret_disc.index, y=ret_disc.values, markers=True,
                      title=T['return_rate_vs_discount'],
                      labels={'x': T['discount_pct'], 'y': T['return_rate_pct']})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ret_del = df.groupby('delivery_days')['is_returned'].mean() * 100
        fig = px.line(x=ret_del.index, y=ret_del.values, markers=True,
                      title=T['return_rate_vs_delivery'],
                      labels={'x': T['delivery_days'], 'y': T['return_rate_pct']})
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        ret_qty = df.groupby('quantity')['is_returned'].mean() * 100
        fig = px.line(x=ret_qty.index, y=ret_qty.values, markers=True,
                      title=T['return_rate_vs_quantity'],
                      labels={'x': T['quantity'], 'y': T['return_rate_pct']})
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 5: Rating Prediction
# ═══════════════════════════════════════════════════════════════════
elif PAGE_KEY == 'nav_rating':
    st.title(T['rating_title'])
    st.markdown(f"*{T['rating_desc']}*")

    st.markdown(f"### {T['model_performance']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(T['r2_score'], rating_metrics['R² Score'])
    with col2:
        st.metric(T['mae'], rating_metrics['MAE'])
    with col3:
        st.metric(T['rmse'], rating_metrics['RMSE'])

    st.info(T['rating_note'])

    st.markdown(f"### {T['rating_distribution']}")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='customer_rating', nbins=31,
                           title=T['rating_distribution'],
                           color_discrete_sequence=['#FFA15A'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        rat_cat = df.groupby('product_category')['customer_rating'].mean().sort_values()
        fig = px.bar(x=rat_cat.values, y=rat_cat.index, orientation='h',
                     title=T['avg_rating_by_category'],
                     color=rat_cat.values, color_continuous_scale='Tealgrn',
                     text=rat_cat.values.round(3))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 6: Interactive Predictor
# ═══════════════════════════════════════════════════════════════════
elif PAGE_KEY == 'nav_predictor':
    st.title(T['predictor_title'])
    st.markdown(f"*{T['predictor_desc']}*")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown(f"### {T['order_details']}")
        with st.form("pred_form"):
            prod_cat = st.selectbox(T['product_category'],
                                    sorted(df['product_category'].unique()))
            region = st.selectbox(T['region'], sorted(df['region'].unique()))
            payment = st.selectbox(T['payment_method'],
                                   sorted(df['payment_method'].unique()))

            col_a, col_b = st.columns(2)
            with col_a:
                prod_price = st.slider(f"{T['product_price']} (USD)", 5.0, 500.0, 250.0, 5.0)
                quantity = st.slider(T['quantity'], 1, 6, 3)
            with col_b:
                discount = st.select_slider(T['discount_pct_label'], [0, 5, 10, 15, 20], 0)
                deliv_days = st.slider(T['delivery_days_label'], 1, 9, 5)

            col_a, col_b = st.columns(2)
            with col_a:
                order_month = st.slider(T['order_month'], 1, 12, 6)
            with col_b:
                day_of_week = st.slider(T['day_of_week'], 0, 6, 3)

            submitted = st.form_submit_button(T['predict_btn'],
                                              use_container_width=True, type="primary")

    with col2:
        if submitted:
            input_data = pd.DataFrame([{
                'product_price': prod_price,
                'quantity': quantity,
                'delivery_days': deliv_days,
                'discount_percent': discount,
                'product_category_enc': label_encoders['product_category'].transform([prod_cat])[0],
                'region_enc': label_encoders['region'].transform([region])[0],
                'payment_method_enc': label_encoders['payment_method'].transform([payment])[0],
                'month': order_month,
                'day_of_week': day_of_week,
            }])

            X_input = input_data[feature_cols].values
            rev_pred = rf_reg.predict(X_input)[0]
            ret_pred = rf_clf.predict(X_input)[0]
            ret_prob = rf_clf.predict_proba(X_input)[0]
            rat_pred = rf_rating.predict(X_input)[0]

            st.markdown(f"### {T['model_predictions']}")

            c1, c2, c3 = st.columns(3)
            with c1:
                base_price = prod_price * quantity
                delta_pct = (rev_pred - base_price) / base_price * 100 if base_price > 0 else 0
                st.metric(T['predicted_revenue_label'], f"{rev_pred:,.2f}",
                          delta=f"{delta_pct:.0f}% {T['vs_base_price']}")
            with c2:
                ret_label = T['likely_return'] if ret_pred == 1 else T['likely_keep']
                st.metric(T['return_prediction'], ret_label,
                          delta=f"{T['return_probability']}: {ret_prob[1]:.1%}")
            with c3:
                st.metric(T['predicted_rating'], f"{rat_pred:.2f}")

            st.markdown("---")
            st.markdown(f"### {T['probability_breakdown']}")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[T['not_returned'], T['returned']],
                                 y=[ret_prob[0], ret_prob[1]],
                                 text=[f'{ret_prob[0]:.1%}', f'{ret_prob[1]:.1%}'],
                                 textposition='auto',
                                 marker_color=['#00CC96', '#EF553B']))
            fig.update_layout(title=T['probability_breakdown'], height=250,
                              yaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"### {T['input_summary']}")
            summary = pd.DataFrame({
                T['feature']: [T['product_category'], T['region'], T['payment_method'],
                               T['product_price'], T['quantity'],
                               T['discount_pct_label'], T['delivery_days_label'],
                               T['order_month'], T['day_of_week']],
                T['value']: [prod_cat, region, payment, f'{prod_price}', quantity,
                             f'{discount}%', deliv_days, order_month, day_of_week]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.markdown(f"### {T['fill_instructions']}")
            st.markdown(f"- **{T['fill_revenue']}**")
            st.markdown(f"- **{T['fill_return']}**")
            st.markdown(f"- **{T['fill_rating']}**")

# ═══════════════════════════════════════════════════════════════════
# PAGE 7: Model Comparison
# ═══════════════════════════════════════════════════════════════════
elif PAGE_KEY == 'nav_comparison':
    st.title(T['comparison_title'])

    st.markdown(f"### {T['all_models']}")

    comp_data = pd.DataFrame({
        T['model']: [T['revenue_model'], T['return_model'], T['rating_title']],
        T['type']: [T['nav_revenue'].split('（')[0], T['nav_return'].split('（')[0],
                     T['nav_rating'].split('（')[0]],
        T['algorithm']: ['Random Forest', 'Random Forest', 'Random Forest'],
        T['key_metric']: ['R² Score', 'ROC-AUC', 'R² Score'],
        T['value']: [reg_metrics['R² Score'], clf_metrics['ROC-AUC'], rating_metrics['R² Score']],
        T['secondary_metric']: ['MAE (USD)', 'F1 Score', 'MAE'],
        T['secondary_metric'].replace('MAE', 'MAE (USD)'): [
            reg_metrics['MAE'], clf_metrics['F1 Score'], rating_metrics['MAE']],
    })
    # Fix: rename column properly
    comp_data = pd.DataFrame({
        T['model']: [T['revenue_model'], T['return_model'], T['rating_desc'].split('(')[0].strip()],
        T['type']: ['Regression', 'Classification', 'Regression'],
        T['algorithm']: ['Random Forest', 'Random Forest', 'Random Forest'],
        T['key_metric']: ['R² Score', 'ROC-AUC', 'R² Score'],
        T['value']: [reg_metrics['R² Score'], clf_metrics['ROC-AUC'], rating_metrics['R² Score']],
        'Secondary': ['MAE (USD)', 'F1 Score', 'MAE'],
        'Secondary Value': [reg_metrics['MAE'], clf_metrics['F1 Score'], rating_metrics['MAE']],
    })
    st.dataframe(comp_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown(f"### {T['top_features']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {T['revenue_model']}")
        fig = px.bar(reg_importance_loc, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     text=reg_importance_loc['Importance'].round(3))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"#### {T['return_model']}")
        fig = px.bar(clf_importance_loc, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Oranges',
                     text=clf_importance_loc['Importance'].round(4))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### {T['key_conclusions']}")
    st.markdown(T['conclusions_table'])

# ── Footer ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{T['tech_stack']}:** Streamlit · scikit-learn · Plotly · Pandas")
st.sidebar.markdown(f"**{T['models_used']}:** Random Forest (Regressor & Classifier)")
