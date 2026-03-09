#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import ssl
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from langchain.schema import BaseMessage, AIMessage, SystemMessage
from langchain.schema.messages import ChatMessage
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIStatusError
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
    before_sleep_log,
)

__package__ = "llms"

import openai
from langchain_core.messages import HumanMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv


# ----------------------------------------------------------------------
# 1️⃣ Unified Retry Decorator
# ----------------------------------------------------------------------
def openai_retry(
        *,
        max_retries: int = 5,
        min_wait: float = 1.0,
        max_wait: float = 60.0,
        logger: Optional[logging.Logger] = None,
) -> Any:
    """
    The Tenacity decorator is used to catch exceptions thrown by the OpenAI SDK and perform exponential backoff retries.

    Parameters
    ----
    max_retries: Maximum number of retries (including the initial call), default is 5.
    min_wait: Number of seconds to wait before the first retry, default is 1 second.
    max_wait: Upper limit for the wait time between retries, default is 60 seconds.
    logger: Optional logger; if provided, logs will be printed before each sleep.
    """
    if logger is None:
        logger = logging.getLogger("RetryChatOpenAI")
        logger.setLevel(logging.INFO)

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=min_wait, max=max_wait) + wait_random(0, 1),
        retry=(retry_if_exception_type(RateLimitError)
               | retry_if_exception_type(APIError)
               | retry_if_exception_type(InternalServerError)
               | retry_if_exception_type(APITimeoutError)
               | retry_if_exception_type(APIConnectionError)
               | retry_if_exception_type(APIStatusError)
               ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


# ----------------------------------------------------------------------
# 2️⃣ Inherit from LangChain's ChatOpenAI and apply a retry decorator to the internal calls.
# ----------------------------------------------------------------------
class RetryChatOpenAI(ChatOpenAI):
    """
    A subclass fully compatible with `langchain.chat_models.ChatOpenAI`,
    with only automatic retry logic added to the `_call` method.

    Example
    ----
    from langchain.schema import HumanMessage
    llm = RetryChatOpenAI(model_name="deepseek-V3-0324", temperature=0.7)
    resp = llm.invoke([HumanMessage(content="Hello!")])
    print(resp.content)
    """

    def __init__(
            self,
            *,
            # ---- LangChain Original Parameters ----
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.0,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            # ---- Retry-related parameters ----
            max_retries: int = 5,
            min_wait: float = 1.0,
            max_wait: float = 60.0,
            # ---- Other OpenAI parameters ----
            **openai_kwargs: Any,
    ) -> None:
        """
        Parameter Description (only new ones listed):
        - **max_retries**: Maximum number of retries (including the initial call), default is 5.
        - **min_wait** / **max_wait**: Minimum/Maximum wait time in seconds for exponential backoff.
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **openai_kwargs,
        )
        # Save retry configuration for use by internal decorators.
        self._max_retries = max_retries
        self._min_wait = min_wait
        self._max_wait = max_wait

        self._logger = logging.getLogger(f"RetryChatOpenAI[{model_name}]")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # 3️⃣ Wrap the actual OpenAI call inside a retry decorator
    # ------------------------------------------------------------------
    @openai_retry(max_retries=5, min_wait=1.0, max_wait=60.0)  # The default values here will be overridden by the configuration in __init__.
    def _completion_with_retry(
            self,
            messages: List[Dict[str, str]],
            **kwargs: Any,
    ) -> openai.ChatCompletion:
        """
        Directly call `openai.ChatCompletion.create` and hand it over to Tenacity for automatic retries.
        This function **does not** require manually catching exceptions; if it ultimately fails, the last exception will be raised.
        """
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 4️⃣ Rewrite the `__call` method of the parent class to use the wrapper function mentioned above.
    # ------------------------------------------------------------------
    def _call(self, messages: List[BaseMessage], run_manager: Any = None) -> str:
        """
        LangChain ultimately calls this function when invoking `invoke` / `__call__`.
        We convert `BaseMessage` into the format required by OpenAI, which is `[{"role": "...", "content": "..."}]`,
        and then pass it to `_completion_with_retry` to complete the actual request.
        """
        # 1) Convert LangChain message objects to OpenAI format
        openai_messages = [
            {
                "role": self._map_role(m),
                "content": m.content,
            }
            for m in messages
        ]

        # 2) Making requests with retries
        response = self._completion_with_retry(openai_messages)

        # 3) Parsing the returned text (compatible with both old and new return structures)
        if response.choices:
            # ChatCompletionMessage has a `message` field (new version) or directly `text` (old version).
            choice = response.choices[0]
            if hasattr(choice, "message"):
                return choice.message["content"]
            else:  # pragma: no cover
                return choice["text"]
        else:  # pragma: no cover
            raise ValueError("OpenAI returned an empty choices list")

    # ------------------------------------------------------------------
    # 5️⃣ Helper function: Map the role of LangChain messages to OpenAI's role
    # ------------------------------------------------------------------
    @staticmethod
    def _map_role(message: BaseMessage) -> str:
        """
        The subclasses of `BaseMessage` used by LangChain:
        - HumanMessage -> "user"
        - AIMessage -> "assistant"
        - SystemMessage -> "system"
        - ChatMessage (role) -> directly use its role
        For other custom messages, please extend them as needed.
        """
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, ChatMessage):
            # ChatMessage comes with a built-in role field, which is returned directly.
            return message.role
        raise TypeError(f"Unsupported message type: {type(message)}")

    # ------------------------------------------------------------------
    # 6️⃣ Allow external parties to modify the retry strategy at any time (optional)
    # ------------------------------------------------------------------
    def set_retry_strategy(
            self,
            *,
            max_retries: Optional[int] = None,
            min_wait: Optional[float] = None,
            max_wait: Optional[float] = None,
    ) -> None:
        """
        Dynamically modify retry configuration. After calling, it will rewrap `_completion_with_retry`.
        """
        if max_retries is not None:
            self._max_retries = max_retries
        if min_wait is not None:
            self._min_wait = min_wait
        if max_wait is not None:
            self._max_wait = max_wait

        # Redecorate once (capture the latest attribute value in the closure)
        self._completion_with_retry = openai_retry(
            max_retries=self._max_retries,
            min_wait=self._min_wait,
            max_wait=self._max_wait,
            logger=self._logger,
        )(self._completion_with_retry.__wrapped__)  # type: ignore


# Add the retry mechanism
def get_model_with_retry(model_name: str,
                         is_outside: bool = False) -> RetryChatOpenAI:
    """Get a ChatOpenAI model instance with retry mechanism.

    Parameters
    ----------
    model_name : str, optional
        The name of the model, you should check the model list inside the company from the link
        https://wiki.huawei.com/domains/88471/wiki/197009/WIKI202502145955402  , by default "deepseek-v3-0324"
    is_outside : bool, optional
        Whether to use an outside model, by default False

    Returns
    -------
    ChatOpenAI
        A ChatOpenAI instance that can be called directly with messages or prompts

    """
    load_dotenv()

    # get the api key and url of the LLM server
    service_api_key = os.getenv("OUT_OPENAI_API_KEY") if is_outside else os.getenv("IN_OPENAI_API_KEY")
    service_api_url = os.getenv("OUT_OPENAI_API_BASE") if is_outside else os.getenv("IN_OPENAI_API_BASE")

    # set the proxy if needed
    proxy_username = os.getenv("USERNAME")
    proxy_pwd = os.getenv("PASSWORD")
    proxy_url = os.getenv("PROXY_URL")

    if is_outside:
        # Customize SSL context to disable DH key checking
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False  # native request interface content
        ssl_ctx.verify_mode = 0  # native request interface content
        ssl_ctx.set_ciphers('DEFAULT@SECLEVEL=1')  # Reduce the security level to allow for smaller DH keys
        client = httpx.Client(
            transport=httpx.HTTPTransport(verify=ssl_ctx, proxy=f"http://{proxy_username}:{proxy_pwd}@{proxy_url}/"), )

        model = RetryChatOpenAI(model=model_name,
                                http_client=client,
                                api_key=service_api_key,
                                base_url=service_api_url,
                                temperature=0.0,
                                max_retries=4,
                                min_wait=2.0,
                                max_wait=20.0, )
    else:
        model = RetryChatOpenAI(model=model_name,
                                api_key=service_api_key,
                                base_url=service_api_url,
                                temperature=0.0,
                                max_retries=4,
                                min_wait=2.0,
                                max_wait=20.0, )
    return model



