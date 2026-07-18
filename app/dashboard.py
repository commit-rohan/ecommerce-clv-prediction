import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Olist CLV & Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------- Custom styling ----------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1d3557 0%, #0b1020 42%, #070b14 100%);
        color: #f8fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.3rem 2.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.20), rgba(59, 130, 246, 0.18));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 14px 38px rgba(0,0,0,0.25);
        margin-bottom: 2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.7rem;
        color: #ffffff;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.08rem;
        margin: 0.6rem 0 0;
    }

    .section-card {
        padding: 1.5rem;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        margin-bottom: 1.3rem;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1rem;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1.1rem;
        background: linear-gradient(90deg, #22c55e, #14b8a6);
        color: white;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: linear-gradient(90deg, #16a34a, #0d9488);
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.75);
        border-radius: 14px;
        padding: 0.8rem;
        border: 1px dashed #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLUMNS = [
    "avg_order_value",
    "n_items",
    "n_distinct_categories",
    "delivery_days",
    "avg_review_score"
]

template_df = pd.DataFrame([{
    "avg_order_value": 150.5,
    "n_items": 2,
    "n_distinct_categories": 1,
    "delivery_days": 8,
    "avg_review_score": 4.2
}])

# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <h1>📊 Olist Customer Intelligence</h1>
    <p>Upload customer data to estimate lifetime value, identify churn risk, and uncover recommended retention actions.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Your CSV must include the following customer-level features.</div>',
        unsafe_allow_html=True
    )

    requirements = pd.DataFrame({
        "Column": REQUIRED_COLUMNS,
        "Description": [
            "Average amount spent per order ($)",
            "Number of items purchased",
            "Number of product categories purchased",
            "Average delivery time in days",
            "Average review score (1–5)"
        ],
        "Example": [150.5, 2, 1, 8, 4.2]
    })
    st.dataframe(requirements, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇ Download sample CSV template",
        data=template_df.to_csv(index=False),
        file_name="template_customers.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload customer data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Choose a CSV file with the required columns to begin prediction.</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader("Customer CSV", type="csv", label_visibility="collapsed")
    st.info("Tip: Download the template if you are unsure about the column format.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Prediction workflow ----------
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Unable to read this file. Please upload a valid CSV.")
        st.stop()

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        st.error(
            "Missing required columns: " + ", ".join(missing_cols) +
            ". Please use the downloadable template."
        )
        st.stop()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    run_predictions = st.button("✨ Run customer predictions", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run_predictions:
        results = []
        errors = 0
        progress = st.progress(0, text="Analyzing customer data...")

        for index, (_, row) in enumerate(df.iterrows()):
            payload = row[REQUIRED_COLUMNS].to_dict()

            try:
                response = requests.post(
                    "https://olist-clv-api.onrender.com/predict",
                    json=payload,
                    timeout=15
                )
                response.raise_for_status()
                results.append(response.json())

            except Exception as error:
                errors += 1
                results.append({
                    "churn_risk": None,
                    "predicted_clv": None,
                    "recommended_action": f"Error: {str(error)}"
                })

            progress.progress(
                (index + 1) / len(df),
                text=f"Analyzing customer {index + 1} of {len(df)}..."
            )

        progress.empty()

        results_df = pd.DataFrame(results)
        final = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        if errors:
            st.warning(f"{errors} row(s) could not be processed. Check numeric values and try again.")

        valid = final.dropna(subset=["predicted_clv"])

        if not valid.empty:
            st.markdown("## Prediction Summary")

            churn_column = valid["churn_risk"].astype(str).str.lower()
            high_risk_count = churn_column.str.contains("high").sum()

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Customers analyzed", len(valid))
            metric2.metric("Average predicted CLV", f"${valid['predicted_clv'].mean():,.2f}")
            metric3.metric("High churn-risk customers", int(high_risk_count))

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Customer prediction results</div>', unsafe_allow_html=True)
        st.dataframe(final, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if not valid.empty:
            chart_df = valid[["predicted_clv"]].copy()
            chart_df.index = [f"Customer {i + 1}" for i in range(len(chart_df))]

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Predicted customer lifetime value</div>', unsafe_allow_html=True)
            st.bar_chart(chart_df, color="#22c55e")
            st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="⬇ Download prediction results",
            data=final.to_csv(index=False),
            file_name="olist_customer_predictions.csv",
            mime="text/csv"
        )