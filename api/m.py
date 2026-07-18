from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    avg_order_value: float
    n_items: int
    n_distinct_categories: int
    delivery_days: float
    avg_review_score: float

class PredictionResponse(BaseModel):
    churn_risk: float
    predicted_clv: float
    recommended_action: str