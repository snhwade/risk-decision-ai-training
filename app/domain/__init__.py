"""领域层：训练任务、交易对手关系图等业务逻辑（不依赖具体框架/技术实现）。

本包对外暴露的端口（Protocol）由基础设施层提供具体实现并在组合根注入：
- TrainingJobRepository：训练任务持久化（对应 ai_training_job 表）
- AlarmNotifier：失败告警通道
- ModelTrainer：实际模型训练算法（可插拔）
"""
