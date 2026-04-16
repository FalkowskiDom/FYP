import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.getenv("DB_PATH", "logs.db")).resolve()

    # HF Token (required for both models)
    hf_token: str = os.getenv("HF_TOKEN", "")
    
    # Qwen model for log explanation
    hf_model_id: str = os.getenv("HF_MODEL_ID", "Biplah/qwen_logdom_3b")
    
    # LogBERT model for anomaly detection
    logbert_model_id: str = os.getenv("LOGBERT_MODEL_ID", "Biplah/logbert_dom")

    default_limit: int = int(os.getenv("DEFAULT_LIMIT", "200"))


settings = Settings()


def require_hf_token() -> None:
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is missing. Set it in environment variables.")