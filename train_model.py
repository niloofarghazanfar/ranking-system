import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# ساخت دیتای fake
np.random.seed(42)

data = pd.DataFrame({
    "rating": np.random.uniform(3, 5, 1000),
    "price": np.random.uniform(20, 100, 1000),
    "completed_tasks": np.random.randint(10, 500, 1000),
    "distance": np.random.uniform(0.5, 10, 1000),
})

# target (شبیه probability booking)
data["target"] = (
    0.6*(data["rating"]/5)
    - 0.3*(data["distance"]/10)
    - 0.2*(data["price"]/100)
    + 0.2*(data["completed_tasks"]/500)
)

X = data[["rating","price","completed_tasks","distance"]]
y = data["target"]

model = lgb.LGBMRegressor()
model.fit(X, y)

# ذخیره مدل
joblib.dump(model, "model.pkl")

print("Model saved ✅")