"""Build the chat model. Switching provider is one line in .env (LLM_PROVIDER)."""
import os

from langchain_core.language_models import BaseChatModel


class ConfigError(RuntimeError):
    pass


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"缺少环境变量 {name}，请在 .env 文件里填写（参考 .env.example）。")
    return value


def build_model(provider: str | None = None) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER", "deepseek")).strip().lower()

    if provider == "deepseek":
        # DeepSeek exposes an OpenAI-compatible API, so ChatOpenAI works with a base_url.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=_require("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0,
            timeout=60,
        )

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            api_key=_require("ANTHROPIC_API_KEY"),
            temperature=0,
            timeout=60,
        )

    raise ConfigError(f"不支持的 LLM_PROVIDER: {provider!r}（可选 deepseek 或 claude）")
