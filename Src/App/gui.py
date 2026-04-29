import json
from typing import Any

import pandas as pd
import requests
import streamlit as st  # type: ignore[import]

st.set_page_config(page_title="AI Log Query Engine", layout="wide")

API_URL = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")
DEFAULT_LIMIT = st.sidebar.slider("Default result limit", min_value=10, max_value=500, value=100, step=10)
DEFAULT_TOKENS = st.sidebar.slider("LLM max new tokens", min_value=64, max_value=1024, value=256, step=64)


def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    # Sends a GET request to the FastAPI backend.
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    # Sends a POST request to the FastAPI backend.
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def logs_to_df(logs: list[dict[str, Any]]) -> pd.DataFrame:
    # Converts log records into a dataframe.
    if not logs:
        return pd.DataFrame()

    preferred_order = [
        "id",
        "ts",
        "event_id",
        "source",
        "user",
        "host",
        "ip",
        "anomaly_score",
        "anomaly_label",
        "message",
        "template",
        "raw",
    ]

    df = pd.DataFrame(logs)

    # Orders the most important columns first.
    ordered = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    return df[ordered]


def render_logs(logs: list[dict[str, Any]], title: str = "Logs") -> None:
    # Displays logs in a table and allows CSV download.
    st.subheader(title)
    df = logs_to_df(logs)

    if df.empty:
        st.info("No logs returned.")
        return

    st.dataframe(df, use_container_width=True, height=420)

    # Adds a download button for the results.
    st.download_button(
        "Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="log_results.csv",
        mime="text/csv",
    )


st.title("AI-Driven Log Analysis Tool")
st.caption("Natural-language querying, event lookup, suspicious log review, and CSV ingestion")

# Creates the main navigation tabs.
query_tab, event_tab, suspicious_tab, ingest_tab, raw_tab = st.tabs(
    ["Natural Language Query", "Event Lookup", "Suspicious Logs", "CSV Ingest", "Raw JSON"]
)

with query_tab:
    st.subheader("Ask a question about the logs")
    query_text = st.text_area(
        "Query",
        value="Show event 4625 logs and summarise anything suspicious.",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        query_limit = st.number_input("Limit", min_value=1, max_value=1000, value=DEFAULT_LIMIT)
    with col2:
        query_validate = st.checkbox("Run validation", value=True)

    if st.button("Run query", type="primary"):
        try:
            # Sends the natural language query to the backend.
            result = api_post(
                "/query",
                {
                    "query": query_text,
                    "limit": int(query_limit),
                    "run_validation": query_validate,
                    "max_new_tokens": int(DEFAULT_TOKENS),
                },
            )

            st.success(f"Mode: {result.get('mode', 'unknown')} | Count: {result.get('count', 0)}")

            # Displays the LLM summary if one is returned.
            if result.get("summary"):
                st.markdown("### LLM Summary")
                st.write(result["summary"])

            # Displays the LLM answer if one is returned.
            if result.get("answer"):
                st.markdown("### LLM Answer")
                st.write(result["answer"])

            render_logs(result.get("logs", []), "Query Results")

            # Saves the response for the Raw JSON tab.
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(str(e))

with event_tab:
    st.subheader("Lookup logs by event ID")

    col1, col2 = st.columns(2)
    with col1:
        event_id = st.number_input("Event ID", min_value=1, value=4625)
    with col2:
        event_validate = st.checkbox("Run validation on event results", value=True, key="event_validate")

    if st.button("Fetch event logs"):
        try:
            # Requests logs that match the selected event ID.
            result = api_get(
                f"/events/{int(event_id)}",
                params={"limit": int(DEFAULT_LIMIT), "run_validation": event_validate},
            )

            st.success(f"Event {result.get('event_id')} | Count: {result.get('count', 0)}")
            render_logs(result.get("logs", []), f"Event {int(event_id)} Results")

            # Saves the response for the Raw JSON tab.
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(str(e))

with suspicious_tab:
    st.subheader("Review suspicious or anomalous logs")
    st.write("This uses the existing /query route with a suspicious/anomaly prompt.")

    suspicious_prompt = st.text_input(
        "Prompt",
        value="Show suspicious logs and summarise why they are suspicious.",
    )

    if st.button("Find suspicious logs"):
        try:
            # Sends a suspicious log query to the backend.
            result = api_post(
                "/query",
                {
                    "query": suspicious_prompt,
                    "limit": int(DEFAULT_LIMIT),
                    "run_validation": True,
                    "max_new_tokens": int(DEFAULT_TOKENS),
                },
            )

            st.success(f"Count: {result.get('count', 0)}")

            # Displays the LLM summary if one is returned.
            if result.get("summary"):
                st.markdown("### LLM Summary")
                st.write(result["summary"])

            render_logs(result.get("logs", []), "Suspicious Results")

            # Saves the response for the Raw JSON tab.
            st.session_state["last_result"] = result
        except Exception as e:
            st.error(str(e))

with ingest_tab:
    st.subheader("Upload CSV and ingest into the database")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        # Reads the uploaded CSV file.
        df = pd.read_csv(uploaded)

        st.write("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Ingest CSV"):
            try:
                lines = []

                # Converts each CSV row into a raw log line.
                for _, row in df.iterrows():
                    raw = row.get("raw") or row.get("message") or json.dumps(row.fillna("").to_dict())
                    lines.append(str(raw))

                # Sends the CSV log lines to the backend.
                result = api_post("/ingest", {"lines": lines})

                st.success(f"Ingested {result.get('ingested', 0)} rows")

                # Saves the response for the Raw JSON tab.
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(str(e))

with raw_tab:
    st.subheader("Last API response")

    # Displays the most recent API response.
    last_result = st.session_state.get("last_result")

    if last_result is None:
        st.info("Run a query, event lookup, suspicious search, or ingestion first.")
    else:
        st.json(last_result)