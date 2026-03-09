import os
from pathlib import Path

import openai
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from .proxied_chatopenai import RobustProxyChatOpenAI
except Exception:  # pragma: no cover
    from proxied_chatopenai import RobustProxyChatOpenAI


def _load_env() -> None:
    if load_dotenv is None:
        return
    llm_dir = Path(__file__).resolve().parent
    parent_dir = llm_dir.parent
    repo_dir = parent_dir.parent
    for env_path in (llm_dir / ".env", parent_dir / ".env", repo_dir / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_env()

def get_llm(model_name: str = "gpt-oss-120b", is_outside: bool = False) -> ChatOpenAI:
    """Obtain a callable llm based on the model_name, by default this model is an inner model.

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

    # get the api key and url of the LLM server
    default_model = os.getenv("AGENT_MODEL") or os.getenv("MODEL") or model_name
    service_api_key = (
        os.getenv("OUT_OPENAI_API_KEY")
        if is_outside
        else (os.getenv("IN_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"))
    )
    service_api_url = (
        os.getenv("OUT_OPENAI_API_BASE")
        if is_outside
        else (os.getenv("IN_OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("API_BASE"))
    )
    target_model = default_model

    if is_outside:
        model = RobustProxyChatOpenAI(model=target_model,
                           api_key=service_api_key,
                           base_url=service_api_url,
                           temperature=0.0,
                           max_retries=5,)
    else:
        model = ChatOpenAI(model=target_model,
                           api_key=service_api_key,
                           base_url=service_api_url,
                           temperature=0.0,
                           max_retries=5,)

    return model


# Added a fallback mechanism
def get_model_with_fallback(model_name: str,
                            fallback_model_name: str,
                            is_outside: bool = False) -> Runnable:
    """Get a ChatOpenAI model instance with fallback mechanism.

    Parameters
    ----------
    model_name : str
        The primary model name to use
    fallback_model_name : str
        The fallback model name to use if the primary model is unavailable
    is_outside : bool, optional
        Whether to use an outside model, by default False

    Returns
    -------
    Runnable
        A Runnable instance configured with fallback mechanism

    """
    fall_back_kwargs = dict(
        fallbacks=[get_llm(fallback_model_name, is_outside)],
        exceptions_to_handle=(openai.APIStatusError,
                              openai.BadRequestError,
                              openai.RateLimitError)
        # Mostly model name error, can not find a model with corresponding name
    )
    return get_llm(model_name, is_outside).with_fallbacks(**fall_back_kwargs)
