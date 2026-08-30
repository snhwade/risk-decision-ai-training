# risk-decision-ai-training

**风控实时决策平台 · AI 训练与在线评分服务**

本仓库是风控实时决策平台的 **AI 旁路增强** 组件：基于 MySQL 历史订单数据进行离线模型训练，提取交易对手关系图特征，并将 AI 指标写入 [risk-decision-services](https://github.com/snhwade/risk-decision-services) 的 `indicator-store-service`，供规则/决策引擎在决策流中引用。

> **不阻塞实时决策链路** — 训练与评分均为异步旁路，决策引擎通过 HTTP 调用在线评分接口。

## 项目定位

| 项 | 说明 |
|---|---|
| 框架 | FastAPI + Uvicorn |
| 端口 | 8000（默认） |
| 数据 | MySQL 历史订单（`risk_decision` 库） |
| 输出 | 指标写入 indicator-store（8084） |
| 架构 | DDD 分层（adapter / domain / infrastructure） |

## 能力概览

### 离线训练

- **监督学习（欺诈评分）**：以 `final_decision`（REJECT=1, PASS=0）为标签，GBDT 训练 `ai_fraud_score`
- **无监督异常检测**：IsolationForest 训练 `ai_anomaly_score`
- **交易对手图分析**：networkx 构建关系图，输出社区规模、环路标记、PageRank 等
- **定时调度**：Cron 表达式配置自动训练任务
- **模型版本管理**：多版本并存，可切换当前生效版本

### 在线评分

- 决策流 **MODEL 节点** 实时调用 `POST /api/v1/ai/score`
- 返回模型分数供 Aviator 规则引用

### 训练产出指标

| 指标名 | 说明 |
|--------|------|
| `ai_fraud_score` | 监督欺诈概率分 |
| `ai_anomaly_score` | 无监督异常分 |
| `ai_cp_degree` / `ai_cp_count` 等 | 交易对手基础图特征 |
| `ai_cp_community_size` | 社区规模 |
| `ai_cp_ring_flag` | 环路标记 |
| `ai_cp_pagerank` | PageRank 中心性 |

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查与配置快照 |
| POST | `/api/v1/ai/training-jobs` | 提交训练任务 `{dataFrom, dataTo}` |
| GET | `/api/v1/ai/training-jobs` | 查询任务列表与状态 |
| POST | `/api/v1/ai/score` | 在线评分（决策流 MODEL 节点） |
| GET | `/api/v1/ai/models` | 模型列表 |
| GET | `/api/v1/ai/models/{modelKind}` | 模型详情 |
| PUT | `/api/v1/ai/models/{modelKind}/current` | 激活指定版本 |
| PUT | `/api/v1/ai/models/{modelKind}` | 更新模型元数据 |
| GET/POST/PUT/DELETE | `/api/v1/ai/training-schedules[/{id}]` | 训练计划 CRUD |
| POST | `/api/v1/ai/training-schedules/{id}/run-now` | 立即触发训练 |

完整 API 文档：启动后访问 `http://localhost:8000/docs`（Swagger UI）

## 技术栈

- **FastAPI** + **Pydantic v2** — REST API 与校验
- **SQLAlchemy** + **PyMySQL** — MySQL 访问
- **pandas** + **scikit-learn** — 特征工程与模型（缺失时纯 Python 降级）
- **networkx** — 交易对手图分析（缺失时纯 Python 降级）
- **httpx** — 调用 indicator-store 写指标
- **croniter** — 定时调度
- **pytest** + **hypothesis** — 测试

## 前置依赖

| 组件 | 说明 |
|------|------|
| MySQL | 历史订单数据，默认 `risk_decision` 库 |
| indicator-store-service | `http://localhost:8084`，接收 AI 指标写入 |
| Python | 3.10+ 推荐 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MYSQL_URL` | — | MySQL 连接串 |
| `INDICATOR_STORE_URL` | `http://localhost:8084` | 指标存储服务地址 |
| `MIN_TRAINING_SAMPLES` | `1000` | 最少训练样本数 |
| `MAX_TRAINING_SECONDS` | `3600` | 单次训练超时（秒） |
| `MODEL_STORE_DIR` | `./models` | 模型文件存储目录 |
| `SCHEDULER_ENABLED` | `true` | 是否启用定时调度 |
| `AUTO_PROMOTE_ON_SAVE` | `false` | 训练完成后是否自动激活新版本 |

## 快速开始

```powershell
git clone https://github.com/snhwade/risk-decision-ai-training.git
cd risk-decision-ai-training

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 配置环境变量（或 .env 文件）
$env:MYSQL_URL = "mysql+pymysql://root:password@localhost:3306/risk_decision"
$env:INDICATOR_STORE_URL = "http://localhost:8084"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 提交训练任务示例

```bash
curl -X POST http://localhost:8000/api/v1/ai/training-jobs \
  -H "Content-Type: application/json" \
  -d '{"dataFrom": "2026-01-01", "dataTo": "2026-06-30"}'
```

### 在线评分示例

```bash
curl -X POST http://localhost:8000/api/v1/ai/score \
  -H "Content-Type: application/json" \
  -d '{"modelKind": "fraud", "features": {"amount": 10000, "merchantId": "M123"}}'
```

## 目录结构

```
app/
├── adapter/           # HTTP 控制器、路由、Schema
├── domain/            # 领域模型与服务
├── infrastructure/    # MySQL、模型存储、指标写入客户端
├── composition.py     # 依赖注入根
└── main.py            # FastAPI 入口
tests/
requirements.txt
```

## 与平台集成

```
MySQL 历史订单
      │
      ▼
 AI Training Service（本仓库）
      │  训练完成
      ▼
 indicator-store-service（写 ai_* 指标）
      │
      ▼
 rule-decision-engine（决策流 MODEL 节点读取 / 调用 /score）
      │
      ▼
 admin-console /ai-training 页面（任务与模型管理）
```

## 关联仓库

| 仓库 | 说明 |
|------|------|
| [risk-decision-services](https://github.com/snhwade/risk-decision-services) | Java 后端（引擎调用评分、BFF 代理 AI API） |
| [risk-decision-admin-console](https://github.com/snhwade/risk-decision-admin-console) | 前端 AI 训练管理页 |
| [risk-decision-commons](https://github.com/snhwade/risk-decision-commons) | 公共 Java 库 |
| [risk-decision-data-engine](https://github.com/snhwade/risk-decision-data-engine) | 数据引擎（旁路计算） |

## 测试

```powershell
pytest tests/ -v
```

## 设计原则

- **旁路增强**：训练失败或超时不会阻断实时决策
- **纯 Python 降级**：scikit-learn / networkx 不可用时仍可运行基础逻辑
- **指标统一出口**：所有 AI 产出通过 indicator-store 写入，与 Flink 累计指标读取路径一致
