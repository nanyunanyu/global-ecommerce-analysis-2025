import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# 设置页面配置
st.set_page_config(
    page_title="电商销售多维分析系统 by 文瑞锋",
    page_icon="https://img.icons8.com/fluent/80/shopping-bag.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式 - 引入更高级的视觉设计
st.markdown("""
    <style>
    /* 引入渐变背景 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 指标卡片玻璃拟态效果 */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* 自定义 Tab 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: none;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    
    /* 标题美化 */
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* 侧边栏美化 */
    .css-1d391kg {
        background-color: #1E3A8A;
    }
    
    /* 按钮美化 */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #1E3A8A;
        color: white;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #3B82F6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_ecommerce_sales_2025.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

@st.cache_data
def perform_clv_analysis(_df):
    if _df.empty:
        return pd.DataFrame()
    df = _df.copy()
    df.loc[df['is_returned'] == 1, 'revenue'] = 0
    snapshot_date = df['order_date'].max() + timedelta(days=1)
    
    # 1. 基础特征提取
    customer_df = df.copy()
    customer_df['recency'] = (snapshot_date - customer_df['order_date']).dt.days
    customer_df['frequency_raw'] = 1
    customer_df = customer_df.rename(columns={'revenue': 'monetary', 'customer_rating': 'rating'})
    customer_df = customer_df.set_index('customer_id')[['recency', 'frequency_raw', 'monetary', 'discount_percent', 'rating']]
    
    # 2. 品类偏好
    category_prefs = pd.get_dummies(df['product_category'], prefix='cat')
    category_prefs.index = df['customer_id']
    
    # 3. 合并特征
    features_df = customer_df.merge(category_prefs, left_index=True, right_index=True)
    
    # 4. K-Means 聚类
    cluster_cols = [c for c in features_df.columns if c != 'frequency_raw']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df[cluster_cols])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # 5. 启发式 CLV (针对无重复购买数据集)
    # 统一使用启发式评分作为演示
    features_df['predictive_clv'] = features_df['monetary'] * (features_df['rating'] / 5)
    features_df['expected_avg_sales'] = features_df['monetary']
    
    return features_df

try:
    df = load_data()

    # --- 侧边栏设计 ---
    st.sidebar.title("控制面板")
    st.sidebar.markdown("---")

    # 日期筛选
    min_date = df['order_date'].min().to_pydatetime()
    max_date = df['order_date'].max().to_pydatetime()
    
    st.sidebar.subheader("📅 时间跨度")
    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        start_date = st.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
    with col_end:
        end_date = st.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)

    # 联动过滤器
    st.sidebar.subheader("维度筛选")
    all_categories = sorted(df['product_category'].unique())
    selected_cats = st.sidebar.multiselect("商品类别", all_categories, default=all_categories)

    all_regions = sorted(df['region'].unique())
    selected_regions = st.sidebar.multiselect("销售地区", all_regions, default=all_regions)

    # 支付方式
    all_payments = sorted(df['payment_method'].unique())
    selected_payments = st.sidebar.multiselect("支付方式", all_payments, default=all_payments)

    # 数据过滤逻辑
    if start_date > end_date:
        st.sidebar.error("错误：开始日期不能晚于结束日期")
        filtered_df = df.iloc[0:0]
    else:
        mask = (
            (df['order_date'].dt.date >= start_date) & 
            (df['order_date'].dt.date <= end_date) &
            (df['product_category'].isin(selected_cats)) &
            (df['region'].isin(selected_regions)) &
            (df['payment_method'].isin(selected_payments))
        )
        filtered_df = df.loc[mask]

    # --- 主界面设计 ---
    st.title("全球电商销售多维分析面板")
    st.info(f"""数据来源：[Kaggle Datasets](https://www.kaggle.com/datasets/emirhanakku/synthetic-e-commerce-sales-dataset-2025)""")
    # AI 智能摘要
    with st.expander("业务洞察报告", expanded=True):
        col_ai1, col_ai2 = st.columns([1, 1])
        with col_ai1:
            total_rev = filtered_df['revenue'].sum()
            avg_rev = filtered_df['revenue'].mean()
            st.markdown(f"""
            **核心结论：**
            -  本周期内总营收达到 **¥{total_rev:,.2f}**，平均单笔订单贡献为 **¥{avg_rev:.2f}**。
            -  **{filtered_df.groupby('region')['revenue'].sum().idxmax()}** 地区贡献了最高的营收额，是当前的核心市场。
            -  **{filtered_df['product_category'].value_counts().idxmax()}** 类目订单量最为密集，市场渗透率高。
            """)
        with col_ai2:
            return_rate = (filtered_df['is_returned'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            rating_avg = filtered_df['customer_rating'].mean()
            st.markdown(f"""
            **风险预警：**
            -  当前整体退货率为 **{return_rate:.1f}%**，{'处于健康范围' if return_rate < 5 else '需要关注供应链质量'}。
            -  客户平均满意度为 **{rating_avg:.2f}/5.0**，满意度{'良好' if rating_avg > 4 else '有待提升'}。
            -  平均物流时效为 **{filtered_df['delivery_days'].mean():.1f}** 天。
            """)

    # KPI 核心指标 - 使用玻璃拟态卡片效果
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("总营收 (Revenue)", f"${total_rev/10000:.1f}W", delta=f"{len(filtered_df)} 订单")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("平均评分 (Rating)", f"{rating_avg:.2f}", delta=f"{rating_avg-3.5:.2f} vs 基准")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("退货率 (Return Rate)", f"{return_rate:.1f}%", delta=f"{-return_rate:.1f}%", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_delivery = filtered_df['delivery_days'].mean()
        st.metric("平均物流时效", f"{avg_delivery:.1f} 天", delta="-0.5 天")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 图表 Tabs
    tab4, tab1, tab2, tab3, tab5 = st.tabs(["客户价值分析", "业绩趋势", "地区与分类", "支付与物流", "原始数据"])

    with tab1:
        st.subheader("营收增长趋势")
        freq = st.radio("时间维度", ["日", "周", "月"], horizontal=True, index=2)
        freq_map = {"日": "D", "周": "W", "月": "ME"}
        ts_data = filtered_df.set_index('order_date').resample(freq_map[freq])['revenue'].sum().reset_index()
        fig_ts = px.line(ts_data, x='order_date', y='revenue', title=f"{freq}度营收走势", markers=True,
                        color_discrete_sequence=['#1E3A8A'])
        fig_ts.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("各地区营收分布")
            region_rev = filtered_df.groupby('region')['revenue'].sum().reset_index()
            fig_region = px.pie(region_rev, values='revenue', names='region', hole=0.5, 
                               title="地区营收占比", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_region.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_region, use_container_width=True)
        with c2:
            st.subheader("品类营收排行")
            cat_rev = filtered_df.groupby('product_category')['revenue'].sum().sort_values(ascending=True).reset_index()
            fig_cat = px.bar(cat_rev, x='revenue', y='product_category', orientation='h', title="品类营收对比", 
                            color='revenue', color_continuous_scale='Blues')
            fig_cat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cat, use_container_width=True)

    with tab3:
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("支付方式偏好")
            pay_counts = filtered_df['payment_method'].value_counts().reset_index()
            fig_pay = px.funnel(pay_counts, x='count', y='payment_method', title="支付方式使用频率",
                               color_discrete_sequence=['#3B82F6'])
            fig_pay.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pay, use_container_width=True)
        with c4:
            st.subheader("物流时效分布")
            fig_delivery = px.histogram(filtered_df, x="delivery_days", color="product_category", 
                                       marginal="box", title="各品类物流时效分布",
                                       color_discrete_sequence=px.colors.qualitative.Safe)
            fig_delivery.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_delivery, use_container_width=True)

    with tab4:
        st.subheader("基于机器学习的客户分群与 CLV 预测")
        
        with st.spinner("正在运行聚类模型与价值评估..."):
            clv_results = perform_clv_analysis(filtered_df)
            
        if clv_results.empty:
            st.warning("当前筛选条件下无数据，无法进行价值分析。")
        else:
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.markdown("#### 1. 客户分群可视化 (K-Means)")
                # 简化版聚类可视化：Recency vs Monetary
                fig_cluster = px.scatter(
                    clv_results, 
                    x="recency", 
                    y="monetary", 
                    color=clv_results["cluster"].astype(str),
                    size="predictive_clv",
                    hover_data=["rating", "discount_percent"],
                    title="客户聚类分布 (气泡大小代表预测 CLV)",
                    labels={"recency": "距今购买天数", "monetary": "消费金额", "color": "分群 ID"}
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
                
            with c2:
                st.markdown("#### 2. 分群画像雷达图")
                # 计算各群体的标准化均值用于雷达图
                cluster_profile = clv_results.groupby('cluster')[['recency', 'monetary', 'rating', 'discount_percent']].mean()
                # 标准化处理
                cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
                
                categories = cluster_profile_norm.columns.tolist()
                fig_radar = go.Figure()

                for i in cluster_profile_norm.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=cluster_profile_norm.loc[i].values,
                        theta=categories,
                        fill='toself',
                        name=f'分群 {i}'
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="不同客群多维特征对比",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # 辅助表格
                st.markdown("**分群核心指标均值**")
                st.dataframe(cluster_profile.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

            st.markdown("---")
            
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("#### 3. 核心客群识别")
                # 识别逻辑
                high_clv_q = clv_results['predictive_clv'].quantile(0.8)
                at_risk_q = clv_results['recency'].quantile(0.7)
                new_q = clv_results['recency'].quantile(0.2)
                
                at_risk_df = clv_results[(clv_results['predictive_clv'] >= high_clv_q) & (clv_results['recency'] >= at_risk_q)]
                potential_new_df = clv_results[(clv_results['predictive_clv'] >= high_clv_q) & (clv_results['recency'] <= new_q)]
                
                st.success(f"**高价值即将流失用户**: {len(at_risk_df)} 人")
                st.info(f"**潜在高净值新用户**: {len(potential_new_df)} 人")
                
                target_group = st.selectbox("查看名单详情", ["高价值即将流失用户", "潜在高净值新用户"])
                if target_group == "高价值即将流失用户":
                    st.dataframe(at_risk_df[['recency', 'monetary', 'rating', 'predictive_clv']].head(100), use_container_width=True)
                else:
                    st.dataframe(potential_new_df[['recency', 'monetary', 'rating', 'predictive_clv']].head(100), use_container_width=True)
                    
            with c4:
                st.markdown("#### 4. 预测性 CLV 分布")
                fig_clv_dist = px.histogram(clv_results, x="predictive_clv", nbins=50, title="预测性客户终身价值分布", 
                                           color_discrete_sequence=['#FF4B4B'])
                st.plotly_chart(fig_clv_dist, use_container_width=True)

            #st.info("💡 **运营建议**: 请根据各分群的偏好（如 Electronics/Fashion）进行个性化召回或留存激励。")

    with tab5:
        st.subheader("筛选数据预览")
        st.dataframe(filtered_df, use_container_width=True)
        
        # 导出功能
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="导出筛选数据为 CSV",
            data=csv,
            file_name=f'sales_export_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

except Exception as e:
    st.error(f"分析系统运行出错: {e}")
