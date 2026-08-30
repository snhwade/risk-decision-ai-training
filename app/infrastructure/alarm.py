"""告警通道（R13.7/13.8：训练失败/指标写入失败时触发告警）。

默认实现基于标准库 logging，将告警写入 WARNING 级别日志，便于被日志/监控系统采集。
生产环境可替换为对接 Micrometer 计数 + 告警通道（如企业 IM/邮件/PagerDuty）的实现。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("ai_training.alarm")


class LoggingAlarmNotifier:
    """基于日志的告警通道实现。"""

    def alarm(self, title: str, detail: str) -> None:
        logger.warning("[AI训练告警] %s | %s", title, detail)
