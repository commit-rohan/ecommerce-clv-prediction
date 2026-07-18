# E-Commerce Customer CLV & Churn Risk Prediction

Predicting a new customer's expected lifetime value and repeat-purchase likelihood using signals available at their first order — built to help an e-commerce marketing team prioritize retention spend and acquisition budget.

## Live Demo

- **API (Swagger UI):** `<your-render-url>/docs`
- **Dashboard:** `<your-streamlit-cloud-url>`

*(Add these once deployed)*

## Problem

Most churn-prediction projects assume customers repeat-purchase often. This dataset doesn't work that way — Olist's real repeat-purchase rate is only ~3%. Rather than force a "will this customer churn in 30 days" framing onto data that doesn't support it, this project reframes the problem as:

> **Predict a new customer's expected lifetime value and likelihood of becoming a repeat buyer, using only signals available at or shortly after their first order.**

This is a more honest fit for the data, and closer to how real e-commerce teams use first-purchase signals to decide acquisition spend per customer segment.

## Dataset

[Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — ~100k orders from 2016–2018, across 8 relational tables (orders, customers, order items, payments, reviews, products, sellers, geolocation).

## Approach

1. **Data merging** — joined 6+ relational tables into a single customer-level feature set, keyed on `customer_unique_id` (not `customer_id`, which is generated per-order and would make every customer look like a one-time buyer).
2. **Feature engineering** — RFM-style features (total spend, average order value, item count, category diversity) plus delivery speed and review score, all computed from signals available near the first order to avoid leakage.
3. **Time-based train/test split** — split by `first_order_date` rather than randomly, to avoid leaking future information into the training set.
4. **Two-stage modeling:**
   - **Stage 1 (classification):** predicts repeat-purchase likelihood — baseline Logistic Regression compared against XGBoost.
   - **Stage 2 (regression):** predicts customer lifetime value (CLV) — XGBoost Regressor.
5. **Experiment tracking** — MLflow used to log parameters and metrics across training runs.
6. **Interpretability** — SHAP values used to explain what drives each prediction (e.g., delivery speed and review score are strong CLV signals).
7. **Deployment** — model served via a FastAPI `/predict` endpoint, containerized with Docker, with a Streamlit dashboard for non-technical use (upload a CSV, get back predictions + recommended actions).

## Key Finding

Repeat purchase rate in this dataset is ~3% — this shaped the entire project framing. Instead of a churn classifier assuming customers return regularly, the project predicts value-at-acquisition, which is what's actually estimable from this data.

## Tech Stack

`pandas` · `scikit-learn` · `xgboost` · `shap` · `mlflow` · `FastAPI` · `Streamlit` · `Docker` · `pytest`

## Project Structure

```
ecommerce-clv/
├── data/
│   ├── raw/              # Olist CSVs (not tracked in git)
│   └── processed/        # merged/engineered feature tables
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── churn_model.pkl
│   └── clv_model.pkl
├── api/
│   ├── main.py            # FastAPI app
│   └── schemas.py         # request/response models
├── app/
│   └── dashboard.py        # Streamlit dashboard
├── tests/
│   └── test_features.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running Locally

**1. Clone and set up environment**
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-clv-prediction.git
cd ecommerce-clv-prediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Run the API**
```bash
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs`

**3. Run the dashboard** (in a separate terminal, with the API still running)
```bash
streamlit run app/dashboard.py
```

**4. Or run everything in Docker**
```bash
docker build -t olist-clv-api .
docker run -p 8000:8000 olist-clv-api
```

**5. Run tests**
```bash
pytest tests/
```

## Sample Prediction

```json
POST /predict
{
  "avg_order_value": 150.5,
  "n_items": 2,
  "n_distinct_categories": 1,
  "delivery_days": 8,
  "avg_review_score": 4.2
}
```
```json
{
  "churn_risk": 0.259,
  "predicted_clv": 298.5,
  "recommended_action": "Healthy — no action needed"
}
```

## What I'd Do With More Time

- Incorporate marketing spend data to compute true ROI per acquisition channel, not just raw CLV
- Add a proper drift-monitoring pipeline to detect when the model's assumptions stop holding
- Try a survival-analysis approach for time-to-next-purchase, as an alternative to the repeat/no-repeat classifier
- Expand the feature set with geographic and seasonal signals

## License

MIT