import ast
import json
import re
from datetime import datetime, timezone  

EVENT_RE = re.compile(r"(?:EventID|EventId|event_id)\s*[:=]\s*(\d+)", re.IGNORECASE)
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+\b")
NUMBER_RE = re.compile(r"\b\d+\b")


def extract_template(message: str) -> str:
    if not message:
        return ""
    message = IP_RE.sub("<IP>", message)
    message = EMAIL_RE.sub("<EMAIL>", message)
    message = NUMBER_RE.sub("<NUM>", message)
    return message


def _build_record(obj: dict, raw_line: str) -> dict:
    event_id = (
        obj.get("EventId")
        or obj.get("EventID")
        or obj.get("event_id")
        or obj.get("EventRecordId")
    )

    ts = obj.get("TimeCreated") or obj.get("timestamp") or obj.get("ts")
    source = (
        obj.get("ProviderName")
        or obj.get("Provider")
        or obj.get("source")
        or obj.get("provider")
    )

    msg = (
        obj.get("Message")
        or obj.get("message")
        or obj.get("MapDescription")
        or obj.get("Payload")
        or ""
    )

    user = (
        obj.get("TargetUserName")
        or obj.get("SubjectUserName")
        or obj.get("UserName")
        or obj.get("user")
    )

    host = obj.get("Computer") or obj.get("host")
    ip = obj.get("IpAddress") or obj.get("RemoteHost") or obj.get("ip")

    if not ts:
        ts = datetime.now(timezone.utc).isoformat()

    try:
        event_id = int(event_id) if event_id is not None else None
    except (TypeError, ValueError):
        event_id = None

    return {
        "ts": ts,
        "event_id": event_id,
        "source": source,
        "message": msg,
        "user": user,
        "host": host,
        "ip": ip,
        "template": extract_template(msg),
        "raw": raw_line,
    }


def parse_line(line: str) -> dict:
    line = line.strip()
    if not line:
        return {}

    if line.startswith("{") and line.endswith("}"):
        try:
            obj = json.loads(line)
            return _build_record(obj, line)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, dict):
                    return _build_record(obj, line)
            except (ValueError, SyntaxError):
                pass

    m = EVENT_RE.search(line)
    event_id = int(m.group(1)) if m else None

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        "source": None,
        "message": line,
        "user": None,
        "host": None,
        "ip": None,
        "template": extract_template(line),
        "raw": line,
    }