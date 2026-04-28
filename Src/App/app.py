from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import re
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from Src.App.config import settings

from .db import init_db
from .parse import parse_line
from .retrieval import (
    insert_logs,
    get_by_event_id,
    get_recent,
    get_suspicious,
    get_by_user,
    get_by_host,
)
from .validator import Validator
from .hf_llm import generate_text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    lines: list[str]


class QueryRequest(BaseModel):
    query: str
    limit: int = 200
    run_validation: bool = True
    max_new_tokens: int = 256


# Real-time ingest state
ingest_task_running = False


def run_realtime_ingest():
    """Background task for real-time Windows event log ingestion"""
    import time
    import requests
    from collections import deque
    
    # Windows-only import with fallback
    try:
        import win32evtlog
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
        logger.warning("win32evtlog not available. Real-time ingestion only works on Windows.")
        return

    if not HAS_WIN32:
        logger.warning("Skipping real-time ingestion - Windows-only feature")
        return

    API = "http://127.0.0.1:8000/ingest"
    SERVER = "localhost"
    LOG_TYPE = "Security"
    
    try:
        hand = win32evtlog.OpenEventLog(SERVER, LOG_TYPE)
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        seen = deque(maxlen=10000)
        
        def format_event(event):
            return str({
                "EventID": event.EventID & 0xFFFF,
                "TimeGenerated": str(event.TimeGenerated),
                "SourceName": event.SourceName,
                "Message": str(event.StringInserts)
            })
        
        logger.info("Real-time ingestion started")
        global ingest_task_running
        ingest_task_running = True
        
        while ingest_task_running:
            try:
                events = win32evtlog.ReadEventLog(hand, flags, 0)
                lines = []
                
                for event in events:
                    key = event.RecordNumber
                    if key in seen:
                        continue
                    seen.append(key)
                    line = format_event(event)
                    lines.append(line)
                
                if lines:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            r = requests.post(API, json={"lines": lines}, timeout=30)
                            r.raise_for_status()
                            logger.info(f"Ingested {len(lines)} logs")
                            break
                        except requests.exceptions.RequestException as e:
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to ingest after {max_retries} attempts: {e}")
                            else:
                                wait_time = 2 ** attempt
                                logger.warning(f"Ingest attempt {attempt + 1} failed, retrying in {wait_time}s...")
                                time.sleep(wait_time)
                
                time.sleep(3)
            except Exception as e:
                logger.error(f"Error in real-time ingest loop: {e}")
                time.sleep(5)
                
    except Exception as e:
        logger.error(f"Failed to start real-time ingestion: {e}")
        ingest_task_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan - startup and shutdown events"""
    logger.info("Starting up AI Log Query Engine...")
    init_db()
    
    # Start real-time ingestion in background
    import threading
    ingest_thread = threading.Thread(target=run_realtime_ingest, daemon=True)
    ingest_thread.start()
    
    yield
    
    # Shutdown
    global ingest_task_running
    ingest_task_running = False
    logger.info("Shutting down...")


app = FastAPI(
    title="AI Log Query Engine",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# Use HF Inference API for LogBERT
# validator = Validator(
#     mode="logbert", 
#     hf_model_id=settings.logbert_model_id
# )
validator = Validator(
    mode="logbert",
    repo_path="Models/logbert"
)

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests with timing"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    return response


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "realtime_ingest_running": ingest_task_running
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Ingest log lines into the database"""
    rows = []
    
    for line in req.lines:
        r = parse_line(line)
        if r:
            rows.append(r)
    
    if rows:
        rows = validator.score(rows)
        insert_logs(rows)
        logger.info(f"Ingested {len(rows)} log entries")
    
    return {"ingested": len(rows), "processed": len(req.lines)}


@app.get("/events/{event_id}")
def events(event_id: int, limit: int = 200, run_validation: bool = True):
    """Get logs by event ID"""
    logs = get_by_event_id(event_id, limit=limit)
    return {"event_id": event_id, "count": len(logs), "logs": logs}


@app.post("/query")
def query(req: QueryRequest):
    """Natural language query endpoint"""
    q = req.query
    q_lower = q.lower()

    # Event ID search
    m = re.search(r"\b(?:event)\s*(\d+)\b", q_lower)
    if m:
        event_id = int(m.group(1))
        logs = get_by_event_id(event_id, limit=req.limit)
        vlogs = logs

        prompt = (
            f"User question: {req.query}\n\n"
            f"Logs:\n{json.dumps(vlogs[:30], indent=2)}\n\n"
            "Summarise and flag anything suspicious."
        )

        summary = generate_text(prompt, max_new_tokens=req.max_new_tokens)

        return {
            "mode": "event",
            "event_id": event_id,
            "count": len(logs),
            "summary": summary,
            "logs": vlogs,
        }

    # Suspicious logs search
    if "suspicious" in q_lower or "anomaly" in q_lower:
        logs = get_suspicious(limit=req.limit)
        prompt = f"Summarise why these logs are suspicious:\n{json.dumps(logs[:30], indent=2)}"
        summary = generate_text(prompt, max_new_tokens=req.max_new_tokens)

        return {
            "mode": "suspicious",
            "count": len(logs),
            "summary": summary,
            "logs": logs,
        }

    # User search
    m = re.search(r"user\s+([^\s]+)", q, re.IGNORECASE)
    if m:
        user = m.group(1)
        logs = get_by_user(user, limit=req.limit)

        return {
            "mode": "user",
            "user": user,
            "count": len(logs),
            "logs": logs,
        }

    # Host search
    m = re.search(r"host\s+([^\s]+)", q, re.IGNORECASE)
    if m:
        host = m.group(1)
        logs = get_by_host(host, limit=req.limit)

        return {
            "mode": "host",
            "host": host,
            "count": len(logs),
            "logs": logs,
        }

    # General query
    logs = get_recent(limit=req.limit)
    prompt = f"User question: {req.query}\n\nLogs:\n{json.dumps(logs[:30], indent=2)}"
    answer = generate_text(prompt, max_new_tokens=req.max_new_tokens)

    return {
        "mode": "general",
        "count": len(logs),
        "answer": answer,
        "logs": logs,
    }