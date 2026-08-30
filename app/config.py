"""AI 训练服务配置（从环境变量读取，全部有默认值便于本地启动）。"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """服务配置。"""

    # MySQL 连接（读取历史交易订单数据）
    mysql_url: str = os.getenv(
        "MYSQL_URL",
        "mysql+pymysql://risk:risk@localhost:3306/risk_decision",
    )
    # 指标存储服务地址（写入交易对手关系指标）
    indicator_store_base_url: str = os.getenv(
        "INDICATOR_STORE_URL", "http://localhost:8084"
    )
    # 训练样本与超时控制（R13.6 / R13.7）
    min_training_samples: int = int(os.getenv("MIN_TRAINING_SAMPLES", "1000"))
    max_training_seconds: int = int(os.getenv("MAX_TRAINING_SECONDS", "3600"))
    # 指标写入重试次数（R13.8）
    indicator_write_max_retries: int = int(os.getenv("INDICATOR_WRITE_MAX_RETRIES", "3"))
    # 模型存储目录（S12.2 模型持久化与版本管理；joblib 落盘，支持训练一次多次评分/回滚）
    model_store_dir: str = os.getenv(
        "MODEL_STORE_DIR",
        os.path.join(os.path.expanduser("~"), ".risk_ai_models"),
    )
    # 后台定时调度（按 cron 自动触发训练）
    scheduler_enabled: bool = os.getenv("SCHEDULER_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    scheduler_poll_seconds: int = int(os.getenv("SCHEDULER_POLL_SECONDS", "60"))
    # 训练成功后是否自动设为 current（IM2：默认关闭，需在模型管理手动启用）
    auto_promote_on_save: bool = os.getenv("AUTO_PROMOTE_ON_SAVE", "false").lower() in (
        "1",
        "true",
        "yes",
    )

def get_settings() -> Settings:
    return Settings()
