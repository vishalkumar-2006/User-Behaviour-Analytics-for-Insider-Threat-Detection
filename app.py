import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Insider Threat Detection", layout="wide")

st.title("🔐 Insider Threat Detection System")
st.write("Upload raw CERT log CSV to predict insider risk.")

# ==========================
# Check required model files
# ==========================
required_files = [
    'rf_insider_model.pkl',
    'rf_scaler.pkl',
    'rf_threshold.pkl',
    'rf_features.pkl'
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ {file} not found. Please train the model first.")
        st.stop()

# ==========================
# Load model artifacts
# ==========================
rf_model     = joblib.load('rf_insider_model.pkl')
scaler       = joblib.load('rf_scaler.pkl')
threshold    = joblib.load('rf_threshold.pkl')
feature_cols = joblib.load('rf_features.pkl')

st.sidebar.header("⚙️ Settings")
st.sidebar.write(f"**Loaded threshold:** `{threshold:.4f}`")

custom_threshold = st.sidebar.slider(
    "Override detection threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(threshold),
    step=0.01,
    help="Lower = more sensitive (more flags)."
)

st.sidebar.write(f"**Active threshold:** `{custom_threshold:.4f}`")
st.sidebar.markdown("---")
st.sidebar.write("**Expected model features:**")
st.sidebar.write(feature_cols)


# ==========================
# Helper: read CERT CSV safely
# ==========================
CERT_COLS = ["activity", "id", "timestamp", "user", "pc",
             "to", "cc", "bcc", "from_addr", "size", "attachments", "content"]

def read_cert_csv(uploaded_file):
    """
    Always reads without header, assigns CERT column names,
    and deduplicates any repeated names so pyarrow never crashes.
    """
    df = pd.read_csv(
        uploaded_file,
        header=None,
        engine="python",
        on_bad_lines="skip",
        dtype=str
    )

    base_names = CERT_COLS[:len(df.columns)]
    while len(base_names) < len(df.columns):
        base_names.append(f"extra_{len(base_names)}")

    seen = {}
    final_cols = []
    for col in base_names:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)

    df.columns = final_cols

    for col in ["size", "attachments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ==========================
# Feature Engineering
# ==========================
def engineer_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts > 0:
        st.warning(f"⚠️ Dropped {bad_ts} rows with unparseable timestamps.")
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        st.error("❌ No valid rows remain after timestamp parsing.")
        st.stop()

    df["hour"]    = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["date"]    = df["timestamp"].dt.date

    df["after_hours"] = ((df["hour"] < 8) | (df["hour"] > 18)).astype(int)
    df["weekend"]     = (df["weekday"] >= 5).astype(int)

    num_days = df["date"].nunique()

    user_features = df.groupby("user").agg(
        logon_total       = ("timestamp",   "count"),
        logon_after_hours = ("after_hours", "sum"),
        logon_weekend     = ("weekend",     "sum"),
        logon_unique_pcs  = ("pc",          "nunique"),
    ).reset_index()

    user_features["logon_after_hours_ratio"] = (
        user_features["logon_after_hours"] / user_features["logon_total"]
    ).fillna(0)

    user_features["logon_weekend_ratio"] = (
        user_features["logon_weekend"] / user_features["logon_total"]
    ).fillna(0)

    user_features["logon_rate_per_day"] = (
        user_features["logon_total"] / (num_days + 1)
    )

    if "activity" in df.columns:
        activity_counts = df.groupby(["user", "activity"]).size().unstack(fill_value=0)
        activity_counts.columns = [f"act_{c.lower().replace(' ', '_')}" for c in activity_counts.columns]
        user_features = user_features.merge(activity_counts.reset_index(), on="user", how="left")

    if "size" in df.columns:
        size_stats = df.groupby("user")["size"].agg(
            email_size_total="sum",
            email_size_max="max",
            email_size_mean="mean",
        ).reset_index()
        user_features = user_features.merge(size_stats, on="user", how="left")

    if "attachments" in df.columns:
        att_stats = df.groupby("user")["attachments"].agg(
            attachments_total="sum",
            attachments_max="max",
        ).reset_index()
        user_features = user_features.merge(att_stats, on="user", how="left")

    for col in feature_cols:
        if col not in user_features.columns:
            user_features[col] = 0

    cols_to_return = [c for c in feature_cols if c in user_features.columns]
    user_features_out = user_features[cols_to_return].fillna(0)

    return user_features, user_features_out


# ==========================
# Prediction
# ==========================
def predict_users(user_df_full, model_input_df, active_threshold):
    scaled = scaler.transform(model_input_df)
    probs  = rf_model.predict_proba(scaled)[:, 1]
    preds  = (probs >= active_threshold).astype(int)

    result = user_df_full[["user"]].copy() if "user" in user_df_full.columns else model_input_df.copy()
    result = result.reset_index(drop=True)
    result["predicted_probability"] = probs
    result["predicted_label"]       = preds
    result["risk_level"] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )
    return result


# ==========================
# File Upload UI
# ==========================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = read_cert_csv(uploaded_file)

    st.write("### Raw Uploaded Data")
    st.write(f"Shape: `{df.shape}`  |  Columns: `{list(df.columns)}`")
    st.write(df.head(10))

    for req in ["timestamp", "user", "pc"]:
        if req not in df.columns:
            st.error(f"❌ Required column `{req}` not found. Found: {list(df.columns)}")
            st.stop()

    with st.spinner("Engineering features..."):
        user_features_full, model_input = engineer_features(df)

    st.write("### Engineered User Features")
    st.write(user_features_full.head(10))

    with st.expander("Feature Statistics (debug)"):
        st.write(model_input.describe())

    with st.spinner("Running predictions..."):
        predictions = predict_users(user_features_full, model_input, custom_threshold)

    st.write("### All Prediction Results")
    st.write(predictions)

    with st.expander("Probability Distribution (debug)"):
        st.write(predictions["predicted_probability"].describe())
        st.bar_chart(
            predictions["predicted_probability"]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )

    high_risk = predictions[predictions["predicted_label"] == 1].sort_values(
        "predicted_probability", ascending=False
    )

    if len(high_risk) == 0:
        st.warning(
            "No high-risk users detected at the current threshold. "
            "Try lowering the threshold slider in the sidebar, "
            "or check the probability distribution above."
        )
    else:
        st.error(f"## 🚨 High Risk Users Detected: {len(high_risk)}")
        st.write(high_risk)

    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions CSV",
        csv_bytes,
        "predictions.csv",
        "text/csv"
    )