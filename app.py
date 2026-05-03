from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.post("/predict")
def predict(data: list):
    scores = [float(x["rating"]) / 5 for x in data]
    return {"scores": scores}