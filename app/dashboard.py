import streamlit as st
import pandas as pd
import requests

st.title("Olist Customer CLV & Churn Risk Dashboard")

REQUIRED_COLUMNS = ['avg_order_value', 'n_items', 'n_distinct_categories',
                     'delivery_days', 'avg_review_score']

st.markdown("""
### How to use this dashboard
Upload a CSV with these exact columns:

| Column | Meaning | Example |
|---|---|---|
| `avg_order_value` | Average amount spent per order ($) | 150.5 |
| `n_items` | Number of items purchased | 2 |
| `n_distinct_categories` | Number of distinct product categories bought | 1 |
| `delivery_days` | Average delivery time (days) | 8 |
| `avg_review_score` | Average review score (1-5) | 4.2 |
""")

# Downloadable template
template_df = pd.DataFrame([{
    'avg_order_value': 150.5,
    'n_items': 2,
    'n_distinct_categories': 1,
    'delivery_days': 8,
    'avg_review_score': 4.2
}])
st.download_button(
    "Download sample CSV template",
    template_df.to_csv(index=False),
    file_name="template_customers.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Upload customer CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        st.error(f"Your CSV is missing these required columns: {', '.join(missing_cols)}. "
                  f"Please download the template above and match its format.")
    else:
        st.write("Preview:", df.head())

        if st.button("Run predictions"):
            results = []
            errors = 0
            for _, row in df.iterrows():
                payload = row[REQUIRED_COLUMNS].to_dict()
                try:
                    resp = requests.post(
                        "https://olist-clv-api.onrender.com/predict",
                        json=payload, timeout=15
                    )
                    resp.raise_for_status()
                    results.append(resp.json())
                except Exception as e:
                    errors += 1
                    results.append({"churn_risk": None, "predicted_clv": None,
                                     "recommended_action": f"Error: {str(e)}"})

            if errors:
                st.warning(f"{errors} row(s) failed — check that all values are numeric.")

            results_df = pd.DataFrame(results)
            final = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            st.dataframe(final, use_container_width=True)

            valid = final.dropna(subset=['predicted_clv'])
            if not valid.empty:
                st.bar_chart(valid['predicted_clv'])