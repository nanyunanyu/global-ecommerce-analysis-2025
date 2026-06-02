# Django + Vue 电商机器学习分析仪表盘实现计划

## 目标

在当前目录的新文件夹 `django_vue_ml_dashboard/` 中搭建 Django + Vue 电商机器学习分析仪表盘。新项目不修改原 Streamlit 文件，支持 CSV 导入 SQLite、后端随机森林模型训练、EDA/ML/预测/数据明细/报告页面和一键启动。

## 规格来源

- `docs/superpowers/specs/2026-05-24-django-vue-ml-dashboard-design.md`

## 阶段状态

| 阶段 | 状态 | 目标 |
|---|---|---|
| 1. 初始化项目结构 | complete | 创建新项目文件夹、后端/前端目录、复制数据文件、基础依赖和启动脚本 |
| 2. 实现 Django 数据层 | complete | 建立 Django 项目、订单模型、CSV 导入命令、SQLite 数据查询 |
| 3. 实现 Django API | complete | 总览、EDA、报告、订单分页筛选导出 API |
| 4. 实现机器学习服务 | complete | 特征工程、3 个 Random Forest 模型、指标数据、预测 API |
| 5. 实现 Vue 前端 | complete | 深色高级主题、大卡片布局、路由、API 调用、图表和表格页面 |
| 6. 联调与验证 | complete | 安装/运行检查、导入数据、启动前后端、浏览器验证关键页面 |
| 7. 收尾文档 | complete | 写使用说明、记录限制和验证结果 |
| 8. 高级经营报告改造 | complete | 详细化分析报告内容并升级报告页视觉样式 |
| 9. Streamlit 分析报告浅色改版 | complete | 将原 Streamlit 分析报告页从深色样式改为行政级清爽蓝浅色设计 |

## 当前阶段

全部阶段已完成

## 决策记录

- 新项目文件夹：`django_vue_ml_dashboard/`。
- 后端：Django + Django REST Framework + pandas + scikit-learn。
- 前端：Vue 3 + Vite + Vue Router + Element Plus + ECharts + Axios。
- 数据：保留 CSV，同时导入 SQLite。
- 主题：深色高端数据驾驶舱，大卡片布局。
- 机器学习范围：后端训练 + 指标展示 + 交互式预测。

## 遇到的错误

| 错误 | 尝试次数 | 解决方案 |
|---|---:|---|
| `train_models` 输出 `R²` 在 Windows GBK 控制台触发 UnicodeEncodeError | 1 | 将管理命令成功信息改为 ASCII `R2` |
| DRF APIClient 使用 `testserver` 导致 DisallowedHost | 1 | 将 `testserver` 加入本地 `ALLOWED_HOSTS` |
