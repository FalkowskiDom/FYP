import time
import requests
from collections import deque

API = "http://127.0.0.1:8000/ingest"
SERVER = "localhost"
LOG_TYPE = "Security"

# Windows-only import with fallback
try:
    import win32evtlog
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("Warning: win32evtlog not available. Real-time ingestion only works on Windows.")

if not HAS_WIN32:
    print("Skipping real-time ingestion - Windows-only feature")
    exit(0)

hand = win32evtlog.OpenEventLog(SERVER, LOG_TYPE)
flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ

# FIXED: Use deque with maxsize to prevent memory leak
seen = deque(maxlen=10000)


def format_event(event):
    return str({
        "EventID": event.EventID & 0xFFFF,
        "TimeGenerated": str(event.TimeGenerated),
        "SourceName": event.SourceName,
        "Message": str(event.StringInserts)
    })


while True:
    events = win32evtlog.ReadEventLog(hand, flags, 0)
    lines = []

    for event in events:
        key = event.RecordNumber
        
        if key in seen:
            continue
        
        seen.append(key)  # FIXED: use append instead of add
        line = format_event(event)
        lines.append(line)

    if lines:
        # FIXED: Add retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.post(API, json={"lines": lines}, timeout=30)
                r.raise_for_status()
                print(f"Ingested {len(lines)} logs")
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Error after {max_retries} attempts:", e)
                else:
                    wait_time = 2 ** attempt
                    print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)

    time.sleep(3)