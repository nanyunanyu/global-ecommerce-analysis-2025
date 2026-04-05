# 2025 电商销售多维分析 Dashboard

本项目针对 `synthetic_ecommerce_sales_2025.csv` 销售数据集进行深度多维分析，使用 Streamlit 构建交互式 Dashboard。

## 📊 核心分析维度
- **业绩趋势**：实时监控营收波动，支持日/周/月多级统计频率。
- **市场分布**：透视不同地区与品类的营收占比与排行。
- **用户行为**：分析支付方式偏好与客户评分质量。
- **运营效能**：监控物流时效分布与退货率核心指标。

## 🚀 快速启动

1. **环境安装**：
   ```bash
   pip install -r requirements.txt
   ```

2. **运行应用**：
   ```bash
   streamlit run dashboard.py
   ```

## 🛠️ Windows 部署建议
请参考项目目录下的 `dashboard.py` 进行生产环境部署：
- **方案一：NSSM 服务化** (推荐用于生产)
- **方案二：Windows 任务计划程序** (简单快捷)
- **方案三：IIS 反向代理** (适用于企业域名访问)
