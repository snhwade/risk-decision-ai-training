"""AI 训练服务入站适配层（adapter）。

承载 FastAPI 路由与请求/响应映射，将外部 REST 请求转换为对领域服务
（`app.domain.training_job.TrainingService`）的调用，并把训练任务记录序列化为
对前端友好的结构（满足 R13.9-13.11）。

DDD 分层约定：
- adapter 仅负责协议转换（HTTP ↔ 领域），不承载业务规则；
- 业务规则（样本量/超时/指标写入）位于 domain 层；
- 控制器（TrainingJobController）刻意与 FastAPI 解耦，便于不依赖 Web 框架进行单元测试。
"""
