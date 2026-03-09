import os
import time
import random
import asyncio
import logging
import ssl
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import openai
from langchain_openai import ChatOpenAI
from openai import APIStatusError, APITimeoutError, RateLimitError
from pydantic import Field

logger = logging.getLogger(__name__)

# a list of proxies
KNOWN_PROXY_URLS = [
"proxyjp.huawei.com",
"proxysa.huawei.com",
"proxyae.huawei.com",
"proxyblr.huawei.com"
"proxyph.huawei.com",
"proxybr.huawei.com",
"proxybh.huawei.com",
"proxyza.huawei.com",
"proxype.huawei.com",
"proxycl.huawei.com",
"proxytr.huawei.com",
"proxyca.huawei.com"
]

def _now() -> float:
    return time.time()

def _exp_backoff(attempt: int, base: float = 1.5, cap: float = 15.0, jitter: float = 0.2) -> float:
    delay = min((base ** attempt), cap)
    j = 1.0 + random.uniform(-jitter, jitter)
    return max(0.0, delay * j)

def _is_retryable_exception(e: BaseException) -> Tuple[bool, str]:
    # OpenAI SDK status error
    if isinstance(e, APIStatusError):
        code = getattr(e, "status_code", None)
        if code in (301, 302, 307, 308):
            return True, "openai_redirect"
        if code in (429, 500, 502, 503, 504):
            return True, "openai_http"
        return False, "openai_status"

    if isinstance(e, (APITimeoutError, RateLimitError)):
        return True, e.__class__.__name__

    # httpx transport/network layer
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError, httpx.ProxyError)):
        return True, e.__class__.__name__

    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code in (301, 302, 307, 308):
            return True, "httpx_redirect"
        if code in (429, 500, 502, 503, 504):
            return True, "httpx_http"
        return False, "httpx_status"

    # 证书失败：B 方案理论上不该再出现；但如果出现也重试换代理
    msg = str(e)
    if "CERTIFICATE_VERIFY_FAILED" in msg or "self-signed certificate" in msg:
        return True, "tls_verify_failed"

    return False, "non_retryable"

def _make_insecure_httpx_clients(proxy_url: str, timeout_s: float, follow_redirects: bool = False) -> Tuple[httpx.Client, httpx.AsyncClient]:
    """
    强制 B 方案：
    - 禁用证书校验（CERT_NONE + verify=False）
    - 关闭 hostname 校验
    - 降低 OpenSSL 安全级别 DEFAULT@SECLEVEL=1
    - 通过 proxy 建立隧道
    """
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    timeout = httpx.Timeout(timeout_s)

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    try:
        ssl_ctx.set_ciphers("DEFAULT@SECLEVEL=1")
    except Exception:
        pass

    # 关键：transport.verify=ssl_ctx + Client.verify=False，双保险锁死“不校验”
    sync_transport = httpx.HTTPTransport(proxy=proxy_url, verify=ssl_ctx, retries=0)
    async_transport = httpx.AsyncHTTPTransport(proxy=proxy_url, verify=ssl_ctx, retries=0)

    sync_client = httpx.Client(
        transport=sync_transport,
        timeout=timeout,
        follow_redirects=follow_redirects,  # 让 302 暴露出来
        limits=limits,
        verify=False,
        headers={"User-Agent": "tsagent-proxy-insecure-b/1.0"},
    )

    async_client = httpx.AsyncClient(
        transport=async_transport,
        timeout=timeout,
        follow_redirects=follow_redirects,
        limits=limits,
        verify=False,
        headers={"User-Agent": "tsagent-proxy-insecure-b/1.0"},
    )

    return sync_client, async_client


@dataclass
class _ProxyState:
    healthy: bool = True
    last_used: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure: Optional[float] = None

def proxy_state_init() -> Dict[str, _ProxyState]:

    proxy_username = os.getenv("USERNAME")
    proxy_pwd = os.getenv("PASSWORD")
    if not proxy_username or not proxy_pwd:
        raise ValueError("proxy_username and proxy_pwd must not be empty")

    proxy_urls = [f"http://{proxy_username}:{proxy_pwd}@{proxy_url}:8080/" for proxy_url in KNOWN_PROXY_URLS]

    return {p: _ProxyState() for p in proxy_urls}


def default_proxy() ->str:
    proxy_username = os.getenv("USERNAME")
    proxy_pwd = os.getenv("PASSWORD")
    if not proxy_username or not proxy_pwd:
        raise ValueError("proxy_username and proxy_pwd must not be empty")

    proxy_urls = [f"http://{proxy_username}:{proxy_pwd}@{proxy_url}:8080/" for proxy_url in KNOWN_PROXY_URLS]

    return proxy_urls[0]

class RobustProxyChatOpenAI(ChatOpenAI):
    """
        Proxy rotation + 302/network error retries
        Note: ChatOpenAI is inherited here to avoid compatibility issues caused by differences in client structure within LangChain/OpenAI SDK.
    """

    unhealthy_after:int = Field(default = 3, description="The unhealthy count for the model")
    proxy_status: Dict[str, _ProxyState] = Field(default_factory = proxy_state_init, description ="The field to store the proxy state")
    current_proxy: Optional[str] = Field(default_factory = default_proxy, description ="The field to store the current proxy")
    verbose: bool = Field(default=False, description="The field to indicate whether we should print log")
    request_timeout: Union[float, tuple[float, float], Any, None] = Field(default=60, alias="timeout")

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

    def _choose_proxy(self) -> str:
        # If the current agent is healthy, continue to use it
        if self.current_proxy and self.proxy_status[self.current_proxy].healthy:
            self.proxy_status[self.current_proxy].last_used = _now()
            return self.current_proxy

        # If there is no current agent or the current agent is not healthy, choose a new healthy agent
        healthy = [p for p, st in self.proxy_status.items() if st.healthy]
        if not healthy:
            if self.verbose:
                logger.warning("No healthy proxies available; resetting all proxies to healthy.")
            for p in self.proxy_urls:
                st = self.proxy_status[p]
                st.healthy = True
                st.consecutive_failures = 0
            healthy = list(self.proxy_urls)

        # consecutive_failures Less is preferred, last_used early is preferred
        healthy.sort(key=lambda p: (self.proxy_status[p].consecutive_failures, self.proxy_status[p].last_used))
        chosen = healthy[0]
        self.proxy_status[chosen].last_used = _now()
        self.current_proxy = chosen
        return chosen

    def _mark_success(self, proxy: str) -> None:
        st = self.proxy_status[proxy]
        st.success_count += 1
        st.consecutive_failures = 0
        st.healthy = True

    def _mark_failure(self, proxy: str) -> None:
        st = self.proxy_status[proxy]
        st.failure_count += 1
        st.consecutive_failures += 1
        st.last_failure = _now()
        if st.consecutive_failures >= self.unhealthy_after:
            st.healthy = False
            if self.verbose:
                logger.warning("Proxy marked unhealthy: %s", proxy.split("@")[-1])

    def _build_llm_for_proxy(self, proxy: str) -> Tuple[httpx.Client, httpx.AsyncClient]:

        client_params: dict = {
            "api_key": (
                self.openai_api_key.get_secret_value() if self.openai_api_key else None
            ),
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        sync_client, async_client = _make_insecure_httpx_clients(
            proxy_url=proxy,
            timeout_s=float(self.request_timeout),
            follow_redirects=False,  # The key: let the 302 be exposed to cut the agent and retry
        )


        #  sync client for the model
        self.http_client = sync_client
        sync_specific = {"http_client": self.http_client}
        self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
        self.client = self.root_client.chat.completions

        #  async client for the model
        self.http_async_client = async_client
        async_specific = {"http_client": self.http_async_client}
        self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            )
        self.async_client = self.root_async_client.chat.completions

        return sync_client, async_client

    def get_proxy_info(self) -> Dict[str, Any]:
        return {
            "total_proxies": len(self.proxy_status.keys()),
            "current_proxy": self.current_proxy.split("@")[-1],
            "proxy_status": {
                p: {
                    "healthy": st.healthy,
                    "last_used": st.last_used,
                    "success_count": st.success_count,
                    "failure_count": st.failure_count,
                    "consecutive_failures": st.consecutive_failures,
                    "last_failure": st.last_failure,
                }
                for p, st in self.proxy_status.items()
            },
        }

    # Message Output for the LLM
    def invoke(self, *args: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            proxy = self._choose_proxy()
            if self.verbose:
                logger.info("Using proxy: %s", proxy.split("@")[-1])

            self._build_llm_for_proxy(proxy)
            try:
                result = super().invoke(*args, **kwargs)
                self._mark_success(proxy)
                return result

            except Exception as e:
                last_exc = e
                should_retry, reason = _is_retryable_exception(e)
                if self.verbose:
                    logger.warning("Attempt %d failed (%s) via proxy %s: %s", attempt + 1, reason, proxy.split("@")[-1], str(e))

                self._mark_failure(proxy)

                if attempt >= self.max_retries or not should_retry:
                    raise

                delay = _exp_backoff(attempt)
                if self.verbose:
                    logger.info("Retrying after %.2fs (attempt %d/%d)...", delay, attempt + 2, self.max_retries + 1)
                time.sleep(delay)

        raise RuntimeError(f"All retries failed. Last error: {last_exc}")

    async def ainvoke(self, *args: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            proxy = self._choose_proxy()
            if self.verbose:
                logger.info("Using proxy: %s", proxy.split("@")[-1])

            self._build_llm_for_proxy(proxy)
            try:
                result = await super().ainvoke(*args, **kwargs)
                self._mark_success(proxy)
                return result

            except Exception as e:
                last_exc = e
                should_retry, reason = _is_retryable_exception(e)
                if self.verbose:
                    logger.warning("Async attempt %d failed (%s) via proxy %s: %s", attempt + 1, reason, proxy.split("@")[-1], str(e))

                self._mark_failure(proxy)

                if attempt >= self.max_retries or not should_retry:
                    raise

                delay = _exp_backoff(attempt)
                if self.verbose:
                    logger.info("Retrying after %.2fs (attempt %d/%d)...", delay, attempt + 2, self.max_retries + 1)
                await asyncio.sleep(delay)

        raise RuntimeError(f"All retries failed. Last error: {last_exc}")
    # Stream output for the LLM
    def stream(self, *args: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            proxy = self._choose_proxy()
            if self.verbose:
                logger.info("Using proxy: %s", proxy.split("@")[-1])

            self._build_llm_for_proxy(proxy)
            try:
                result = super().stream(*args, **kwargs)
                self._mark_success(proxy)
                return result

            except Exception as e:
                last_exc = e
                should_retry, reason = _is_retryable_exception(e)
                if self.verbose:
                    logger.warning("Attempt %d failed (%s) via proxy %s: %s", attempt + 1, reason, proxy.split("@")[-1], str(e))

                self._mark_failure(proxy)

                if attempt >= self.max_retries or not should_retry:
                    raise

                delay = _exp_backoff(attempt)
                if self.verbose:
                    logger.info("Retrying after %.2fs (attempt %d/%d)...", delay, attempt + 2, self.max_retries + 1)
                time.sleep(delay)

        raise RuntimeError(f"All retries failed. Last error: {last_exc}")

    async def astream(self, *args: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            proxy = self._choose_proxy()
            if self.verbose:
                logger.info("Using proxy: %s", proxy.split("@")[-1])

            self._build_llm_for_proxy(proxy)
            try:
                result = await super().astream(*args, **kwargs)
                self._mark_success(proxy)
                return result

            except Exception as e:
                last_exc = e
                should_retry, reason = _is_retryable_exception(e)
                if self.verbose:
                    logger.warning("Async attempt %d failed (%s) via proxy %s: %s", attempt + 1, reason, proxy.split("@")[-1], str(e))

                self._mark_failure(proxy)

                if attempt >= self.max_retries or not should_retry:
                    raise

                delay = _exp_backoff(attempt)
                if self.verbose:
                    logger.info("Retrying after %.2fs (attempt %d/%d)...", delay, attempt + 2, self.max_retries + 1)
                await asyncio.sleep(delay)

        raise RuntimeError(f"All retries failed. Last error: {last_exc}")

    # batch output for the LLM
    def batch(self, *args: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            proxy = self._choose_proxy()
            if self.verbose:
                logger.info("Using proxy: %s", proxy.split("@")[-1])

            self._build_llm_for_proxy(proxy)
            try:
                result = super().batch(*args, **kwargs)
                self._mark_success(proxy)
                return result

            except Exception as e:
                last_exc = e
                should_retry, reason = _is_retryable_exception(e)
                if self.verbose:
                    logger.warning("Attempt %d failed (%s) via proxy %s: %s", attempt + 1, reason, proxy.split("@")[-1], str(e))

                self._mark_failure(proxy)

                if attempt >= self.max_retries or not should_retry:
                    raise

                delay = _exp_backoff(attempt)
                if self.verbose:
                    logger.info("Retrying after %.2fs (attempt %d/%d)...", delay, attempt + 2, self.max_retries + 1)
                time.sleep(delay)

        raise RuntimeError(f"All retries failed. Last error: {last_exc}")

    async def abatch(self, *args: Any, **kwargs: Any):
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries + 1):
            proxy = self._choose_proxy()
            if self.verbose:
                logger.info("Using proxy: %s", proxy.split("@")[-1])

            self._build_llm_for_proxy(proxy)
            try:
                result = await super().abatch(*args, **kwargs)
                self._mark_success(proxy)
                return result

            except Exception as e:
                last_exc = e
                should_retry, reason = _is_retryable_exception(e)
                if self.verbose:
                    logger.warning("Async attempt %d failed (%s) via proxy %s: %s", attempt + 1, reason, proxy.split("@")[-1], str(e))

                self._mark_failure(proxy)

                if attempt >= self.max_retries or not should_retry:
                    raise

                delay = _exp_backoff(attempt)
                if self.verbose:
                    logger.info("Retrying after %.2fs (attempt %d/%d)...", delay, attempt + 2, self.max_retries + 1)
                await asyncio.sleep(delay)


        raise RuntimeError(f"All retries failed. Last error: {last_exc}")