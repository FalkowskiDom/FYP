from .db import get_conn

MAX_BATCH_SIZE = 1000


def insert_logs(rows: list[dict]):
    # Inserts a batch of parsed logs into the database.
    if len(rows) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(rows)} exceeds maximum of {MAX_BATCH_SIZE}")

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO logs (
                ts, event_id, source, message,
                user, host, ip, template,
                anomaly_score, anomaly_label, raw
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.get("ts"),
                    r.get("event_id"),
                    r.get("source"),
                    r.get("message"),
                    r.get("user"),
                    r.get("host"),
                    r.get("ip"),
                    r.get("template"),
                    r.get("anomaly_score"),
                    r.get("anomaly_label"),
                    r.get("raw"),
                )
                for r in rows
            ],
        )


def get_by_event_id(event_id: int, limit: int = 200):
    # Gets logs that match a specific event ID.
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM logs WHERE event_id = ? ORDER BY id DESC LIMIT ?",
            (event_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def get_recent(limit: int = 200):
    # Gets the most recent logs from the database.
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_suspicious(threshold: float = 0.8, limit: int = 200):
    # Gets logs with an anomaly score above the threshold.
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT * FROM logs
            WHERE anomaly_score >= ?
            ORDER BY anomaly_score DESC, id DESC
            LIMIT ?
            """,
            (threshold, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def get_by_user(user: str, limit: int = 200):
    # Gets logs that match a specific username.
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT * FROM logs
            WHERE LOWER(COALESCE(user, '')) = LOWER(?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (user, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def get_by_host(host: str, limit: int = 200):
    # Gets logs that match a specific host name.
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT * FROM logs
            WHERE LOWER(COALESCE(host, '')) = LOWER(?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (host, limit),
        )
        return [dict(r) for r in cur.fetchall()]