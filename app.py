from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.metrics import ndcg_score

app = FastAPI()

# load trained model
model = joblib.load("model.pkl")


# -----------------------------
# Data Model
# -----------------------------
class Tasker(BaseModel):
    rating: float
    price: float
    completed_tasks: int
    distance: float


# -----------------------------
# Ranking API
# -----------------------------
@app.post("/rank")
def rank(taskers: list[Tasker]):

    features = []

    for t in taskers:
        features.append([
            t.rating,
            t.price,
            t.completed_tasks,
            t.distance
        ])

    features = np.array(features)

    # ML prediction
    scores = model.predict(features)

    results = []

    for i, t in enumerate(taskers):

        base_score = scores[i]

        # fairness boost
        fairness = 0

        if t.completed_tasks < 50:
            fairness = 0.1

        final_score = base_score + fairness

        results.append({
            "tasker": t.dict(),
            "score": float(final_score),
            "base_score": float(base_score),
            "fairness_boost": fairness
        })

    # Sort ranking
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # -----------------------------
    # NDCG
    # -----------------------------
    try:

        y_true = np.array([[3, 2, 1][:len(scores)]])
        y_pred = np.array([scores[:len(y_true[0])]])

        ndcg = ndcg_score(y_true, y_pred)

    except Exception:
        ndcg = 0.0

    return {
        "ranking": results,
        "ndcg": float(ndcg)
    }