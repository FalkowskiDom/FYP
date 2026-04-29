import math
import re
import logging
from collections import defaultdict
from pathlib import Path
from Src.App.config import settings

import torch
from huggingface_hub import InferenceClient
from transformers import BertForMaskedLM, BertTokenizerFast

logger = logging.getLogger(__name__)

SUSPICIOUS = re.compile(r"(failed|error|denied|malware|attack|unauthorized)", re.IGNORECASE)


class BaseValidator:
    def score(self, logs: list[dict]) -> list[dict]:
        # Defines the scoring method that child validators must implement.
        raise NotImplementedError


SUSPICIOUS = re.compile(
    r"(failed|failure|error|denied|malware|attack|unauthorized|invalid|blocked)",
    re.IGNORECASE
)


class RegexValidator(BaseValidator):
    def score(self, logs: list[dict]) -> list[dict]:
        # Scores logs using simple suspicious keyword matching.
        out = []

        for r in logs:
            msg = f"{r.get('message') or ''} {r.get('raw') or ''}"
            score = 0.85 if SUSPICIOUS.search(msg) else 0.15

            r2 = dict(r)
            r2["anomaly_score"] = score

            if score >= 0.9:
                label = "anomaly"
            elif score >= 0.75:
                label = "suspicious"
            else:
                label = "normal"

            r2["anomaly_label"] = label
            out.append(r2)

        return out


class LogBERTValidator(BaseValidator):
    _model_cache = None
    
    def __init__(
        self,
        repo_path: str = None, 
        hf_model_id: str = None,
        window_size: int = 20,
        threshold: float = 0.95,
        group_by: str = "host",
        fallback_to_regex: bool = True,
    ):
        # Sets up the LogBERT validator settings.
        self.window_size = window_size
        self.threshold = threshold
        self.group_by = group_by
        self.fallback_to_regex = fallback_to_regex
        
        # Decides whether to use the Hugging Face API or a local model.
        self.use_hf_api = hf_model_id is not None
        self.hf_model_id = hf_model_id
        self.hf_token = None
        
        if self.use_hf_api and self.hf_token:
            logger.info(f"Using HF Inference API for LogBERT: {hf_model_id}")
            self._init_hf_api()
        else:
            logger.info("Using local LogBERT model")
            self._init_local(repo_path)

    def _init_hf_api(self):
        """Initialises the Hugging Face Inference API client."""
        try:
            from huggingface_hub import InferenceClient

            self.hf_client = InferenceClient(
                model=self.hf_model_id,
                token=self.hf_token
            )

            self.device = "api"
            self.tokenizer = None
            self.model = None
        except Exception as e:
            logger.error(f"Failed to init HF API, falling back to regex: {e}")
            self.use_hf_api = False

    def _init_local(self, repo_path: str):
        """Initialises the local LogBERT model."""
        if repo_path is None:
            logger.warning("No repo_path provided for local LogBERT")
            self.device = "cpu"
            return
            
        self.repo_path = Path(repo_path)
        self.model_path = self.repo_path / "model"
        self.vocab_path = self.repo_path / "vocab.json"

        if not self.model_path.exists():
            raise RuntimeError(f"LogBERT model folder not found at: {self.model_path}")

        if not self.vocab_path.exists():
            raise RuntimeError(
                f"Template vocab.json not found at: {self.vocab_path}. "
            )

        # Loads the template vocabulary used by LogBERT.
        import json
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.template_vocab = json.load(f)

        # Converts vocabulary IDs back into LogBERT event tokens.
        self.id_to_token = {
            int(v): f"E{int(v)}"
            for k, v in self.template_vocab.items()
            if k not in ("", "<UNK>") and isinstance(v, int)
        }

        self.unk_id = int(self.template_vocab.get("<UNK>", 1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loads the model once and reuses it from cache.
        if LogBERTValidator._model_cache is None:
            self.tokenizer = BertTokenizerFast.from_pretrained(str(self.model_path))
            self.model = BertForMaskedLM.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()

            LogBERTValidator._model_cache = {
                "tokenizer": self.tokenizer,
                "model": self.model
            }
        else:
            self.tokenizer = LogBERTValidator._model_cache["tokenizer"]
            self.model = LogBERTValidator._model_cache["model"]

        self.regex_fallback = RegexValidator()

    def _template_to_event_id(self, template: str) -> int:
        # Converts a log template into its vocabulary ID.
        if not template:
            return self.unk_id if hasattr(self, "unk_id") else 1

        return int(self.template_vocab.get(template, self.unk_id if hasattr(self, "unk_id") else 1))

    def _event_id_to_token(self, event_id: int) -> str:
        # Converts a vocabulary ID into a LogBERT token.
        if hasattr(self, "id_to_token"):
            return self.id_to_token.get(int(event_id), "<UNK>")

        return f"E{event_id}"

    def _build_sequence_tokens(self, rows: list[dict]) -> list[str]:
        # Builds a sequence of LogBERT tokens from log rows.
        tokens = []

        for row in rows:
            template = row.get("template") or row.get("message") or ""
            event_id = self._template_to_event_id(template)
            tokens.append(self._event_id_to_token(event_id))

        return tokens

    def _masked_token_loss_api(self, tokens: list[str], mask_index: int) -> float:
        """Uses the Hugging Face API to estimate masked token loss."""
        try:
            sequence = " ".join(tokens)

            # Masks one token in the sequence.
            masked_tokens = tokens.copy()
            masked_tokens[mask_index] = "[MASK]"
            masked_sequence = " ".join(masked_tokens)
            
            # Gets the model prediction for the masked token.
            response = self.hf_client.fill_mask(masked_sequence)
            
            # Converts prediction confidence into a loss value.
            if response and len(response) > 0:
                top_score = response[0].get("score", 0.5)
                loss = -math.log(max(top_score, 0.01))
                return loss

            return 1.0
        except Exception as e:
            logger.error(f"HF API error in masked_token_loss: {e}")
            return 1.0

    def _masked_token_loss_local(self, tokens: list[str], mask_index: int) -> float:
        """Calculates masked token loss using the local LogBERT model."""
        encoded = self.tokenizer(
            " ".join(tokens),
            return_tensors="pt",
            truncation=True,
            max_length=max(8, self.window_size + 2),
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        labels = input_ids.clone()

        target_pos = mask_index + 1

        if target_pos >= input_ids.shape[1] - 1:
            return 0.0

        # Masks the target token and asks the model to predict it.
        original_token_id = input_ids[0, target_pos].item()
        input_ids[0, target_pos] = self.tokenizer.mask_token_id
        labels[:] = -100
        labels[0, target_pos] = original_token_id

        # Runs the model without training it.
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return float(outputs.loss.item())

    def _masked_token_loss(self, tokens: list[str], mask_index: int) -> float:
        # Chooses API-based or local masked token loss.
        if self.use_hf_api and self.device == "api":
            return self._masked_token_loss_api(tokens, mask_index)

        return self._masked_token_loss_local(tokens, mask_index)

    def _loss_to_score(self, loss_value: float) -> float:
        # Converts model loss into an anomaly score between 0 and 1.
        score = 1.0 - math.exp(-loss_value)
        return max(0.0, min(1.0, score))

    def _score_group(self, rows: list[dict]) -> list[dict]:
        # Scores a group of logs using a sliding window.
        if not rows:
            return []

        rows_sorted = sorted(rows, key=lambda r: str(r.get("ts") or ""))
        out = []

        for i, row in enumerate(rows_sorted):
            start = max(0, i - self.window_size + 1)
            window_rows = rows_sorted[start:i + 1]

            # Uses regex scoring when there are not enough logs for LogBERT.
            if len(window_rows) < 2:
                if self.fallback_to_regex:
                    regex_val = RegexValidator()
                    out.extend(regex_val.score([row]))
                else:
                    r2 = dict(row)
                    r2["anomaly_score"] = 0.0
                    r2["anomaly_label"] = "normal"
                    out.append(r2)
                continue

            # Builds the sequence and scores the newest log.
            tokens = self._build_sequence_tokens(window_rows)
            loss_value = self._masked_token_loss(tokens, len(tokens) - 1)
            score = self._loss_to_score(loss_value)

            r2 = dict(row)
            r2["anomaly_score"] = round(score, 4)

            if score >= 0.95:
                r2["anomaly_label"] = "anomaly"
            elif score >= 0.80:
                r2["anomaly_label"] = "suspicious"
            else:
                r2["anomaly_label"] = "normal"

            out.append(r2)

        return out

    def score(self, logs: list[dict]) -> list[dict]:
        # Groups logs and applies LogBERT scoring to each group.
        if not logs:
            return []

        grouped = defaultdict(list)

        for row in logs:
            group_value = row.get(self.group_by)

            if group_value in (None, "", "unknown_host", "unknown_user"):
                group_value = "__ungrouped__"

            grouped[str(group_value)].append(row)

        out = []

        for _, group_rows in grouped.items():
            out.extend(self._score_group(group_rows))

        return out


class Validator:
    def __init__(self, mode: str = "regex", repo_path: str = None, hf_model_id: str = None):
        # Selects which validator implementation to use.
        if mode == "regex":
            self.impl = RegexValidator()
        elif mode == "logbert":
            self.impl = LogBERTValidator(repo_path=repo_path, hf_model_id=hf_model_id)
        else:
            raise ValueError(f"Unsupported validator mode: {mode}")

    def score(self, logs: list[dict]) -> list[dict]:
        # Scores logs using the selected validator.
        return self.impl.score(logs)