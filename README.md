## ML2 Assignment 1 - Task 3 (CI Quality Gate)

### Run locally
pip install -r requirements.txt
python -m pytest -q

### What the test does
- Loads model from `model/final_model.joblib`
- Loads evaluation data from `data/day_2011.csv`
- Computes RMSE
- Fails if RMSE > gate_ratio * baseline_rmse
