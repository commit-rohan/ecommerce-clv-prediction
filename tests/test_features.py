import pandas as pd
import joblib

def test_models_load():
    """Confirm saved model files load without error."""
    churn_model = joblib.load("src/churn_model.pkl")
    clv_model = joblib.load("src/clv_model.pkl")
    assert churn_model is not None
    assert clv_model is not None

def test_prediction_shape():
    """Confirm a prediction returns a single float, not an array or error."""
    clv_model = joblib.load("src/clv_model.pkl")
    sample = pd.DataFrame([{
        'avg_order_value': 150.5,
        'n_items': 2,
        'n_distinct_categories': 1,
        'delivery_days': 8,
        'avg_review_score': 4.2
    }])
    prediction = clv_model.predict(sample)
    assert len(prediction) == 1
    assert isinstance(float(prediction[0]), float)

def test_clv_prediction_is_reasonable():
    """Sanity check: predicted CLV shouldn't be negative."""
    clv_model = joblib.load("src/clv_model.pkl")
    sample = pd.DataFrame([{
        'avg_order_value': 150.5,
        'n_items': 2,
        'n_distinct_categories': 1,
        'delivery_days': 8,
        'avg_review_score': 4.2
    }])
    prediction = clv_model.predict(sample)[0]
    assert prediction >= 0