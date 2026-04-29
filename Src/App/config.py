import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Stores the database file path.
    db_path: Path = Path(os.getenv("DB_PATH", "logs.db")).resolve()

    # Stores the Hugging Face API token.
    hf_token: str = os.getenv("HF_TOKEN", "")
    
    # Stores the Qwen model ID used for log explanations.
    hf_model_id: str = os.getenv("HF_MODEL_ID", "Biplah/qwen_logdom_3b")
    
    # Stores the LogBERT model ID used for anomaly detection.
    logbert_model_id: str = os.getenv("LOGBERT_MODEL_ID", "Biplah/logbert_dom")

    # Stores the default number of logs to return.
    default_limit: int = int(os.getenv("DEFAULT_LIMIT", "200"))


settings = Settings()


def require_hf_token() -> None:
    # Checks that the Hugging Face token is available.
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is missing. Set it in environment variables.")