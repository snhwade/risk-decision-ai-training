"""交易对手关系指标写入指标存储（基础设施实现，R13.2 / R13.8）。

通过 REST 接口将交易对手关系指标写入 indicator-store-service。写入端口
（`app.domain.counterparty.IndicatorWriter`）约定：写入失败时抛出异常，由领域层
`write_metrics_with_retry` 据 R13.8 进行最多 N 次重试与失败告警；本实现仅负责单次写入。

HTTP 客户端使用开源库 httpx（PyPI 公共 registry）。为保持 domain 层与单元测试不强依赖
网络与 httpx，本模块在构造时才延迟导入 httpx；单元测试可注入内存/mock 替身替换本实现。

指标存储写入契约（与 indicator-store-service 切片模型对齐：refName/dimensionKey/sliceTs/value）：
    POST {base_url}/api/v1/indicators/{refName}
    body: { "dimensionKey": str, "sliceTs": int, "value": float, "source": "AI" }
非 2xx 响应视为写入失败并抛出异常。
"""

from __future__ import annotations

from app.domain.counterparty import CounterpartyMetric


class IndicatorStoreWriteError(RuntimeError):
    """指标存储写入失败异常（携带可读原因，供重试与告警记录）。"""


class HttpxIndicatorWriter:
    """基于 httpx 的指标写入器（IndicatorWriter 端口实现）。

    Args:
        base_url: indicator-store-service 基地址（如 http://localhost:8084）。
        timeout_seconds: 单次写入请求超时（秒），避免写入长时间阻塞。
        client: 可选注入的 httpx.Client（便于复用连接与测试）；缺省时按需创建。
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 2.0,
        client: object | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = client

    def write(self, metric: CounterpartyMetric) -> None:
        url = f"{self._base_url}/api/v1/indicators/{metric.ref_name}"
        payload = {
            "dimensionKey": metric.dimension_key,
            "sliceTs": metric.slice_ts,
            "value": metric.value,
            "source": "AI",
        }
        try:
            if self._client is not None:
                response = self._client.post(url, json=payload, timeout=self._timeout_seconds)
            else:
                import httpx  # 延迟导入，避免测试环境强依赖

                response = httpx.post(url, json=payload, timeout=self._timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - 网络层任意异常统一转写入失败
            raise IndicatorStoreWriteError(
                f"写入指标 {metric.ref_name}/{metric.dimension_key} 网络异常：{exc}"
            ) from exc

        status_code = getattr(response, "status_code", None)
        if status_code is None or not 200 <= int(status_code) < 300:
            raise IndicatorStoreWriteError(
                f"写入指标 {metric.ref_name}/{metric.dimension_key} 失败，HTTP 状态码：{status_code}"
            )
