import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FedEx Shipment ID Matcher", layout="wide")

st.title("FedEx ETA ↔ Raw Data Matcher (CSV only)")
st.write(
    "Upload your **FedEx raw data (CSV)** and the **ETA test file (CSV)**. "
    "The app matches Movement.ID to the first 8 characters of Order number, compares scheduled/actual arrival "
    "dates, and returns the matching Shipment ID."
)

# ----------------------------
# Helpers
# ----------------------------
def canon(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower()).strip()

def guess_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    col_map = {canon(c): c for c in df.columns}

    for cand in candidates:
        key = canon(cand)
        if key in col_map:
            return col_map[key]

    for cand in candidates:
        key = canon(cand)
        for c in df.columns:
            if key and key in canon(c):
                return c
    return None

def clean_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)  # remove Excel-like numeric artifacts
    return s

def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def make_time_key(dt: pd.Series, mode: str) -> pd.Series:
    if mode == "Date only":
        return dt.dt.normalize()
    return dt.dt.floor("min")

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    # dtype=str so IDs don't get mangled, low_memory=False for big CSV stability
    return pd.read_csv(io.BytesIO(file_bytes), dtype=str, low_memory=False)

def run_matching(
    raw: pd.DataFrame,
    eta: pd.DataFrame,
    prefix_len: int,
    match_mode: str,
    raw_order_col: str,
    raw_planned_col: str,
    raw_actual_col: str,
    raw_ship_col: str,
    eta_move_col: str,
    eta_sched_col: str,
    eta_act_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    raw = raw.copy()
    eta = eta.copy()

    # IDs
    raw[raw_order_col] = clean_id_series(raw[raw_order_col])
    eta[eta_move_col] = clean_id_series(eta[eta_move_col])

    raw["_order_prefix"] = raw[raw_order_col].str[:prefix_len]
    eta["_move_prefix"] = eta[eta_move_col].str[:prefix_len]

    # Datetimes
    raw_planned_dt = parse_dt(raw[raw_planned_col])
    raw_actual_dt = parse_dt(raw[raw_actual_col])

    eta_sched_dt = parse_dt(eta[eta_sched_col])
    eta_act_dt = parse_dt(eta[eta_act_col])

    raw["_planned_key"] = make_time_key(raw_planned_dt, match_mode)
    raw["_actual_key"] = make_time_key(raw_actual_dt, match_mode)

    eta["_sched_key"] = make_time_key(eta_sched_dt, match_mode)
    eta["_eta_actual_key"] = make_time_key(eta_act_dt, match_mode)

    # Dedup raw keys to prevent exploding matches
    raw_strict = raw[["_order_prefix", "_planned_key", "_actual_key", raw_ship_col]].dropna(
        subset=["_order_prefix", "_planned_key", "_actual_key"]
    ).drop_duplicates(subset=["_order_prefix", "_planned_key", "_actual_key"], keep="first")

    raw_planned = raw[["_order_prefix", "_planned_key", raw_ship_col]].dropna(
        subset=["_order_prefix", "_planned_key"]
    ).drop_duplicates(subset=["_order_prefix", "_planned_key"], keep="first")

    # Strict: prefix + scheduled + actual
    strict_merged = eta.merge(
        raw_strict,
        left_on=["_move_prefix", "_sched_key", "_eta_actual_key"],
        right_on=["_order_prefix", "_planned_key", "_actual_key"],
        how="left",
    )

    # Planned-only: prefix + scheduled
    planned_merged = eta.merge(
        raw_planned,
        left_on=["_move_prefix", "_sched_key"],
        right_on=["_order_prefix", "_planned_key"],
        how="left",
    )

    eta_out = eta.copy()
    eta_out["Shipment ID (strict match)"] = strict_merged[raw_ship_col]
    eta_out["Shipment ID (planned match)"] = planned_merged[raw_ship_col]
    eta_out["Shipment ID (best)"] = eta_out["Shipment ID (strict match)"].fillna(
        eta_out["Shipment ID (planned match)"]
    )

    eta_out["Match type"] = np.select(
        [
            eta_out["Shipment ID (strict match)"].notna(),
            eta_out["Shipment ID (planned match)"].notna(),
        ],
        ["strict (prefix + scheduled + actual)", "planned-only (prefix + scheduled)"],
        default="no match",
    )

    mapping = eta_out[[eta_move_col, "Shipment ID (best)", "Match type"]].copy()
    mapping = mapping.rename(columns={eta_move_col: "Movement.ID"})

    metrics = {
        "eta_rows": len(eta_out),
        "strict_matches": int(eta_out["Shipment ID (strict match)"].notna().sum()),
        "planned_matches": int(eta_out["Shipment ID (planned match)"].notna().sum()),
        "best_matches": int(eta_out["Shipment ID (best)"].notna().sum()),
        "no_matches": int((eta_out["Shipment ID (best)"].isna()).sum()),
    }

    return eta_out, mapping, metrics


# ----------------------------
# Upload UI
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    raw_file = st.file_uploader("Upload **FedEx raw data** (CSV)", type=["csv"])
with col2:
    eta_file = st.file_uploader("Upload **ETA test file** (CSV)", type=["csv"])

prefix_len = st.number_input(
    "Prefix length to compare (Movement.ID vs Order number prefix)",
    min_value=1, max_value=50, value=8
)

match_mode = st.radio(
    "Match scheduled/actual arrival by:",
    options=["Date only", "Date & time (minute)"],
    index=0,
    help="Default is Date only because requirement said 'if the date matches'."
)

if not raw_file or not eta_file:
    st.info("Upload both files to continue.")
    st.stop()

raw_bytes = raw_file.getvalue()
eta_bytes = eta_file.getvalue()

with st.spinner("Loading CSVs..."):
    df_raw = load_csv(raw_bytes)
    df_eta = load_csv(eta_bytes)

st.subheader("Preview")
p1, p2 = st.columns(2)
with p1:
    st.caption("Raw (first 200 rows)")
    st.dataframe(df_raw.head(200), use_container_width=True)
with p2:
    st.caption("ETA test (first 200 rows)")
    st.dataframe(df_eta.head(200), use_container_width=True)

# ----------------------------
# Column mapping
# ----------------------------
st.subheader("Column mapping (auto-detected — you can override)")

raw_order_guess = guess_col(df_raw, ["Order number", "Order Number", "OrderNumber"])
raw_planned_guess = guess_col(df_raw, ["Destination initial planned arrival time", "Destination planned arrival time"])
raw_actual_guess = guess_col(df_raw, ["Destination actual arrival time", "Destination Actual Arrival Time"])
raw_ship_guess = guess_col(df_raw, ["Shipment ID", "ShipmentId", "Shipment Identifier", "Shipment Identifier"])

eta_move_guess = guess_col(df_eta, ["Movement.ID", "Movement ID", "Movement Id", "MovementID"])
eta_sched_guess = guess_col(df_eta, ["Scheduled Arrival Date/Time", "Scheduled Arrival", "Scheduled arrival date/time"])
eta_act_guess = guess_col(df_eta, ["Actual Arrival Date/Time", "Actual Arrival", "Actual arrival date/time"])

with st.expander("Show / edit column mapping", expanded=True):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**FedEx raw data (CSV)**")
        raw_order_col = st.selectbox(
            "Order number column",
            df_raw.columns,
            index=df_raw.columns.get_loc(raw_order_guess) if raw_order_guess in df_raw.columns else 0
        )
        raw_planned_col = st.selectbox(
            "Destination initial planned arrival time column",
            df_raw.columns,
            index=df_raw.columns.get_loc(raw_planned_guess) if raw_planned_guess in df_raw.columns else 0
        )
        raw_actual_col = st.selectbox(
            "Destination actual arrival time column",
            df_raw.columns,
            index=df_raw.columns.get_loc(raw_actual_guess) if raw_actual_guess in df_raw.columns else 0
        )
        raw_ship_col = st.selectbox(
            "Shipment ID column",
            df_raw.columns,
            index=df_raw.columns.get_loc(raw_ship_guess) if raw_ship_guess in df_raw.columns else 0
        )

    with c2:
        st.markdown("**ETA test file (CSV)**")
        eta_move_col = st.selectbox(
            "Movement.ID column",
            df_eta.columns,
            index=df_eta.columns.get_loc(eta_move_guess) if eta_move_guess in df_eta.columns else 0
        )
        eta_sched_col = st.selectbox(
            "Scheduled Arrival Date/Time column",
            df_eta.columns,
            index=df_eta.columns.get_loc(eta_sched_guess) if eta_sched_guess in df_eta.columns else 0
        )
        eta_act_col = st.selectbox(
            "Actual Arrival Date/Time column",
            df_eta.columns,
            index=df_eta.columns.get_loc(eta_act_guess) if eta_act_guess in df_eta.columns else 0
        )

# ----------------------------
# Run
# ----------------------------
run = st.button("Run matching", type="primary")

if run:
    with st.spinner("Matching..."):
        eta_out, mapping_out, metrics = run_matching(
            raw=df_raw,
            eta=df_eta,
            prefix_len=int(prefix_len),
            match_mode=match_mode,
            raw_order_col=raw_order_col,
            raw_planned_col=raw_planned_col,
            raw_actual_col=raw_actual_col,
            raw_ship_col=raw_ship_col,
            eta_move_col=eta_move_col,
            eta_sched_col=eta_sched_col,
            eta_act_col=eta_act_col,
        )

    st.success("Done.")

    st.subheader("Results summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ETA rows", metrics["eta_rows"])
    m2.metric("Strict matches", metrics["strict_matches"])
    m3.metric("Planned matches", metrics["planned_matches"])
    m4.metric("Best matches", metrics["best_matches"])
    m5.metric("No matches", metrics["no_matches"])

    st.subheader("Output preview")
    st.dataframe(eta_out.head(200), use_container_width=True)

    full_csv = eta_out.to_csv(index=False).encode("utf-8")
    map_csv = mapping_out.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download FULL output (ETA + Shipment IDs) CSV",
        data=full_csv,
        file_name="eta_test_with_shipment_id.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download Movement.ID → Shipment ID mapping CSV",
        data=map_csv,
        file_name="movement_to_shipment_id.csv",
        mime="text/csv",
    )
else:
    st.info("Click **Run matching** to generate the output CSVs.")
