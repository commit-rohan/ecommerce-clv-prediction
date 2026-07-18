import streamlit as st
import pandas as pd
import requests
st.title("Olist Customer CLV & Churn Risk Dashboard")
uploaded = st.file_uploader("Upload customer CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    if st.button("Run predictions"):
        results = []
        for _, row in df.iterrows():
            payload = row[['avg_order_value','n_items','n_distinct_categories',
            'delivery_days','avg_review_score']].to_dict()
            resp = requests.post("http://localhost:8000/predict", json=payload)
            results.append(resp.json())
    results_df = pd.DataFrame(results)
    final = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    st.dataframe(final)
    st.bar_chart(final['predicted_clv'])