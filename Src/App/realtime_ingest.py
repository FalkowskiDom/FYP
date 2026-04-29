import time
import requests
from collections import deque

API = "http://127.0.0.1:8000/ingest"
SERVER = "localhost"
LOG_TYPE = "Security"

# Tries to import the Windows event log library.
try:
    import win32evtlog
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("Warning: win32evtlog not available.")

# Stops the script if Windows event logs are not available.
if not HAS_WIN32:
    print("Skipping real-time ingestion ")
    exit(0)

# Opens the Windows Security event log.
hand = win32evtlog.OpenEventLog(SERVER, LOG_TYPE)

# Reads the newest log entries first.
flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ

# Stores recent record numbers to avoid duplicate logs.
seen = deque(maxlen=10000)


def format_event(event):
    # Converts a Windows event into a dictionary-like string.
    return str({
        "EventID": event.EventID & 0xFFFF,
        "TimeGenerated": str(event.TimeGenerated),
        "SourceName": event.SourceName,
        "Message": str(event.StringInserts)
    })


while True:
    # Reads Windows event logs.
    events = win32evtlog.ReadEventLog(hand, flags, 0)
    lines = []

    # Formats new logs and skips duplicates.
    for event in events:
        key = event.RecordNumber
        
        if key in seen:
            continue
        
        seen.append(key)  
        line = format_event(event)
        lines.append(line)

    # Sends new logs to the API.
    if lines:
        max_retries = 3

        # Retries the request if it fails.
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

    # Waits before checking for new logs again.
    time.sleep(3)