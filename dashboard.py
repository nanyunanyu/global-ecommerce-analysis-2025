import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 设置页面配置
st.set_page_config(
    page_title="全球电商销售多维分析",
    page_icon="https://img.icons8.com/fluent/80/shopping-bag.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_ecommerce_sales_2025.csv")
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

try:
    df = load_data()

    # --- 侧边栏设计 ---
    st.sidebar.title("控制面板")
    st.sidebar.markdown("---")

    # 日期筛选
    min_date = df['order_date'].min().to_pydatetime()
    max_date = df['order_date'].max().to_pydatetime()
    
    st.sidebar.subheader("时间跨度")
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
    st.title("全球电商销售多维分析 Dashboard")
    st.info(f"""当前分析数据量：{len(filtered_df):,} 条记录  
            数据来源：[Kaggle Datasets](https://www.kaggle.com/datasets/emirhanakku/synthetic-e-commerce-sales-dataset-2025)""")

    # KPI 核心指标
    m1, m2, m3, m4 = st.columns(4)
    total_rev = filtered_df['revenue'].sum()
    avg_rating = filtered_df['customer_rating'].mean()
    return_rate = (filtered_df['is_returned'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    avg_delivery = filtered_df['delivery_days'].mean()

    m1.metric("总营收 (Revenue)", f"¥{total_rev:,.0f}")
    m2.metric("平均评分 (Rating)", f"{avg_rating:.2f}分")
    m3.metric("退货率 (Return Rate)", f"{return_rate:.1f}%")
    m4.metric("平均物流时效", f"{avg_delivery:.1f} 天")

    st.markdown("---")

    # 图表 Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["业绩趋势", "地区与分类", "支付与物流", "原始数据"])

    with tab1:
        st.subheader("营收增长趋势")
        freq = st.radio("时间维度", ["日", "周", "月"], horizontal=True, index=2)
        freq_map = {"日": "D", "周": "W", "月": "ME"}
        ts_data = filtered_df.set_index('order_date').resample(freq_map[freq])['revenue'].sum().reset_index()
        fig_ts = px.line(ts_data, x='order_date', y='revenue', title=f"{freq}度营收走势", markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("各地区营收分布")
            region_rev = filtered_df.groupby('region')['revenue'].sum().reset_index()
            fig_region = px.pie(region_rev, values='revenue', names='region', hole=0.4, title="地区营收占比")
            st.plotly_chart(fig_region, use_container_width=True)
        with c2:
            st.subheader("品类营收排行")
            cat_rev = filtered_df.groupby('product_category')['revenue'].sum().sort_values(ascending=True).reset_index()
            fig_cat = px.bar(cat_rev, x='revenue', y='product_category', orientation='h', title="品类营收对比", color='revenue')
            st.plotly_chart(fig_cat, use_container_width=True)

    with tab3:
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("支付方式偏好")
            pay_counts = filtered_df['payment_method'].value_counts().reset_index()
            fig_pay = px.funnel(pay_counts, x='count', y='payment_method', title="支付方式使用频率")
            st.plotly_chart(fig_pay, use_container_width=True)
        with c4:
            st.subheader("物流时效分布")
            fig_delivery = px.histogram(filtered_df, x="delivery_days", color="product_category", marginal="box", title="各品类物流时效分布")
            st.plotly_chart(fig_delivery, use_container_width=True)

    with tab4:
        st.subheader("筛选数据预览")
        st.dataframe(filtered_df, use_container_width=True)
        
        # 导出功能
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 导出筛选数据为 CSV",
            data=csv,
            file_name=f'sales_export_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

except Exception as e:
    st.error(f"分析系统运行出错: {e}")
