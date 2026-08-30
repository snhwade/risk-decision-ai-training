"""超时保护执行器（R13.7：超过最长训练时长则终止训练）。

由于 CPU 密集型训练难以在纯 Python 中被强制抢占，这里采用「工作线程 + 主线程等待」
的实现：在独立线程中执行训练函数，主线程最多等待 timeout_seconds 秒；超时则向上抛出
内置 TimeoutError，由领域服务据此记录失败并告警。

说明：
- 工作线程在超时后不会被强制杀死（Python 无安全的线程强杀机制），但训练函数的结果会被
  丢弃且不会写入任何指标，符合 R13.7「不写入任何交易对手关系指标」的约束。生产实现可进一步
  将训练放入独立子进程并在超时时终止该进程以彻底回收资源。
- 该实现不引入任何第三方依赖，仅使用标准库 threading。
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")


class ThreadTimeoutGuard:
    """基于工作线程的超时保护执行器。"""

    def run(self, func: Callable[[], T], timeout_seconds: float) -> T:
        """在 timeout_seconds 秒内执行 func；超时抛出 TimeoutError，func 内部异常原样抛出。"""
        result: list[T] = []
        error: list[BaseException] = []

        def _worker() -> None:
            try:
                result.append(func())
            except BaseException as exc:  # noqa: BLE001 - 捕获并转交主线程重新抛出
                error.append(exc)

        worker = threading.Thread(target=_worker, name="ai-training-worker", daemon=True)
        worker.start()
        worker.join(timeout_seconds)

        if worker.is_alive():
            # 超时：训练仍在运行，主线程放弃等待并报告超时
            raise TimeoutError(f"训练执行超过 {timeout_seconds} 秒")
        if error:
            raise error[0]
        return result[0]
