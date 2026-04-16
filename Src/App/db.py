import sqlite3
from pathlib import Path
from .config import settings

DB_PATH = str(settings.db_path)

REQUIRED_COLUMNS = {
    "ts": "TEXT",
    "event_id": "INTEGER",
    "source": "TEXT",
    "message": "TEXT",
    "user": "TEXT",
    "host": "TEXT",
    "ip": "TEXT",
    "template": "TEXT",
    "anomaly_score": "REAL",
    "anomaly_label": "TEXT",
    "raw": "TEXT",
}


def get_conn():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30.0,
        check_same_thread=False  # For FastAPI async compatibility
    )
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # Validate column types to prevent injection
    allowed_types = {"TEXT", "INTEGER", "REAL", "BLOB", "NULL"}
    
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """)

        cur = conn.execute("PRAGMA table_info(logs)")
        existing_columns = {row["name"] for row in cur.fetchall()}

        for column_name, column_type in REQUIRED_COLUMNS.items():
            if column_name not in existing_columns:
                if column_type.upper() not in allowed_types:
                    raise ValueError(f"Invalid column type: {column_type}")
                conn.execute(f"ALTER TABLE logs ADD COLUMN {column_name} {column_type}")

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_id ON logs(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON logs(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON logs(user)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_host ON logs(host)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly ON logs(anomaly_score)")
        
        conn.commit() 