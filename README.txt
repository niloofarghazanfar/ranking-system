# Tasker Ranking System 🚀

## 📌 Problem
Ranking taskers effectively is challenging because multiple objectives must be balanced, including user conversion (booking), revenue, and fairness for new taskers. A naive approach may favor only highly rated or experienced taskers, limiting marketplace growth.

## ⚙️ Approach
We designed a ranking system combining feature engineering and machine learning. A baseline heuristic model (based on rating, distance, and experience) is compared against an ML-based ranking model. Multi-objective scoring is applied to balance performance and fairness.

## 🤖 Model
We use a LightGBM ranking model (LambdaMART) trained on features such as:
- Rating
- Price
- Distance
- Completed tasks

The model learns to rank taskers based on their likelihood of being selected.

## 📊 Metrics
We evaluate the system using:

- **NDCG@K**: Measures ranking quality by prioritizing relevant taskers at the top.
- **CTR (Click-Through Rate)**: Measures user engagement.
- **Revenue**: Evaluates business impact.

## 🚀 How to Run

```bash
pip install -r requirements.txt
uvicorn app:app --reload