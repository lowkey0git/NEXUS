"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def _bool(v: str | None, default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = Path("./data")
    provider: str = "anthropic"
    model: str = "claude-sonnet-5-5"
    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    tool_output_chars: int = 12000
    native_web_search: bool = True
    max_steps: int = 12
    max_tokens: int = 4096
    tool_timeout: int = 60
    approval_timeout: int = 900
    max_upload_mb: int = 15
    allow_registration: bool = True
    session_days: int = 14
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_starttls: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(Path(".env"))
        e = os.environ.get
        provider = e("ASSISTANT_PROVIDER", "anthropic")
        default_models = {"anthropic": "claude-sonnet-5-5", "groq": "llama-3.3-70b-versatile", "openai_compat": "gpt-4o-mini"}
        return cls(
            data_dir=Path(e("ASSISTANT_DATA_DIR", "./data")),
            provider=provider,
            model=e("ASSISTANT_MODEL") or default_models.get(provider, ""),
            anthropic_api_key=e("ANTHROPIC_API_KEY") or None,
            anthropic_base_url=e("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            openai_api_key=e("OPENAI_API_KEY") or None,
            openai_base_url=e("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            groq_api_key=e("GROQ_API_KEY") or None,
            groq_base_url=e("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            tool_output_chars=int(e("ASSISTANT_TOOL_OUTPUT_CHARS", "12000")),
            native_web_search=_bool(e("ASSISTANT_NATIVE_WEB_SEARCH"), True),
            max_steps=int(e("ASSISTANT_MAX_STEPS", "12")),
            tool_timeout=int(e("ASSISTANT_TOOL_TIMEOUT", "60")),
            approval_timeout=int(e("ASSISTANT_APPROVAL_TIMEOUT", "900")),
            max_upload_mb=int(e("ASSISTANT_MAX_UPLOAD_MB", "15")),
            allow_registration=_bool(e("ASSISTANT_ALLOW_REGISTRATION"), True),
            smtp_host=e("SMTP_HOST") or None,
            smtp_port=int(e("SMTP_PORT", "587")),
            smtp_user=e("SMTP_USER") or None,
            smtp_password=e("SMTP_PASSWORD") or None,
            smtp_from=e("SMTP_FROM") or None,
            smtp_starttls=_bool(e("SMTP_STARTTLS"), True),
        )

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_from)
