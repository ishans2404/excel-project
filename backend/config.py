from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI
import os


class Settings(BaseSettings):
    # Provider: "groq" | "deepseek" | "openai" | "ollama"
    MODEL_PROVIDER: str = "ollama"

    # ── Groq ──────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "qwen/qwen3-32b"
    GROQ_PLANNER_MODEL: str = "qwen/qwen3-32b"   # Fast planner

    # ── DeepSeek (production, has R1 thinking) ────────────────────────
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-reasoner"           # R1 with thinking
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_PLANNER_MODEL: str = "deepseek-chat"       # V3, fast for planning

    # ── OpenAI ────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_PLANNER_MODEL: str = "gpt-4o-mini"

    # ── Ollama (local LLM) ────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Recommended models for data analysis (install with: ollama pull <model>):
    #   qwen2.5:14b        — best for complex analysis and structured output
    #   qwen2.5-coder:14b  — good for code generation
    #   llama3.1:8b        — faster, less capable
    #   mistral-nemo:12b   — good balance
    OLLAMA_MODEL: str = "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0"
    OLLAMA_PLANNER_MODEL: str = "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0"

    # ── Storage ───────────────────────────────────────────────────────
    UPLOAD_DIR: str = "/tmp/analytics_uploads"
    REPORTS_DIR: str = "/tmp/analytics_reports"
    MAX_FILE_SIZE_MB: int = 200
    SESSION_TTL_HOURS: int = 24

    # ── CORS ──────────────────────────────────────────────────────────
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "https://localhost:3000",
    ]

    LANGGRAPH_RECURSION_LIMIT: int = 60
    CODE_EXEC_TIMEOUT_SECONDS: int = 60

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_llm(self, temperature: float = 0.05, streaming: bool = True, for_planning: bool = False):
        """Return the configured LLM. for_planning=True uses a faster/cheaper model."""
        provider = self.MODEL_PROVIDER

        if provider == "groq":
            model = self.GROQ_PLANNER_MODEL if for_planning else self.GROQ_MODEL
            return ChatOpenAI(
                api_key=self.GROQ_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming,
                max_tokens=8000,
            )

        elif provider == "deepseek":
            model = self.DEEPSEEK_PLANNER_MODEL if for_planning else self.DEEPSEEK_MODEL
            return ChatOpenAI(
                api_key=self.DEEPSEEK_API_KEY,
                base_url=self.DEEPSEEK_BASE_URL,
                model=model,
                temperature=temperature,
                streaming=streaming,
                max_tokens=8000,
            )

        elif provider == "openai":
            model = self.OPENAI_PLANNER_MODEL if for_planning else self.OPENAI_MODEL
            return ChatOpenAI(
                api_key=self.OPENAI_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming,
                max_tokens=8000,
            )

        elif provider == "ollama":
            model = self.OLLAMA_PLANNER_MODEL if for_planning else self.OLLAMA_MODEL
            return ChatOpenAI(
                model=model,
                base_url=self.OLLAMA_BASE_URL,
                temperature=temperature,
                num_predict=6000,
                # Ollama doesn't support streaming=False well for structured output
            )

        else:
            raise ValueError(f"Unknown MODEL_PROVIDER: '{provider}'. Valid: groq | deepseek | openai | ollama")


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)