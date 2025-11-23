# Predictive Maintenance for Factory Equipment

This project is a hands-on predictive maintenance workflow I built for my INFD 615 course. The goal was simple: use real sensor data to predict which machines are likely to fail, then simulate how a maintenance team could monitor those risks in something close to a real-world dashboard.

---

## 🛠 Project Overview

- **Problem:** Unplanned equipment failures are expensive. The idea is to flag “at-risk” machines early so maintenance can act before things break.
- **Data:** 10,000+ records of machine readings, including:
  - Air and process temperatures  
  - Rotational speed (rpm)  
  - Torque  
  - Tool wear (minutes)  
  - Categorical machine type  
  - Binary failure target (failure vs. no failure)

The dataset is highly imbalanced (only about 3.4% of rows are actual failures), so a big part of the project was choosing metrics and models that don’t get fooled by high accuracy on “no failure.”

---

## 🔍 Feature Engineering & Prep

I focused on:

- Selecting key numeric features:  
  `Air temperature [K]`, `Process temperature [K]`,  
  `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- Encoding the categorical `Type` variable with one-hot encoding
- Train/test split (80/20) with a fixed random seed for reproducibility
- Using a `Pipeline` with preprocessing + model so everything is clean and reusable

I also explicitly tracked class imbalance and evaluated metrics like recall, precision, F1, and ROC-AUC instead of just raw accuracy.

---

## 🤖 Models Used

I compared two tree-based classifiers:

1. **Decision Tree**
   - Intuitive, easy to explain
   - Good starting baseline model
2. **Random Forest**
   - Ensemble of many decision trees
   - Usually more stable and higher performing
   - Better at capturing non-linear patterns and interactions

I then did GridSearchCV to tune hyperparameters, especially for the Decision Tree, using recall on the failure class** as the scoring metric. The idea was to prioritize catching actual failures over squeezing out the last 0.1% of accuracy.

---

## 📈 Key Results

### Baseline Models

- **Decision Tree (baseline)**
  - Accuracy: ~97.9%  
  - Failure-class precision: 0.71  
  - Failure-class recall: 0.62  

- **Random Forest (baseline)**
  - Accuracy: ~98.1%  
  - Failure-class precision: 0.94  
  - Failure-class recall: 0.47  

Even though the Random Forest had slightly higher accuracy, the Decision Tree gave a stronger balance between correctly catching failures and staying somewhat interpretable.

---

### Tuned Decision Tree (chosen model)

After hyperparameter tuning with GridSearchCV:

- **Best params:**  
  `max_depth = 5`, `min_samples_leaf = 4`, `min_samples_split = 2`
- **Performance on test set:**
  - Accuracy: **0.97**
  - Failure-class precision: **0.78**
  - Failure-class recall: **0.31**
  - ROC-AUC: **0.93**

The tuned model became more conservative about flagging failures, which improved overall stability and ROC-AUC but lowered recall. In a real deployment, this trade-off would be tuned based on how costly missed failures are vs. false alarms.

---

## 📊 “Dashboard” & Alert Logic

To simulate deployment and monitoring, I:

- Used the tuned Decision Tree to generate **failure probabilities** for each machine.
- Built a simple “dashboard-style” histogram showing:
  - The distribution of predicted failure probabilities
  - A red vertical line at a chosen **alert threshold** (e.g., 0.70)
- Implemented an **alert system**:
  - Machines with `predicted_failure_prob >= 0.70` are flagged
  - Those alerts are written out to `maintenance_alerts.csv` with a machine ID and probability

At a threshold of **0.70**, the system:

- Triggered alerts for **~1.35%** of the machines (27 out of 2000 in the test set)
- Achieved **precision ≈ 0.78** and **recall ≈ 0.31** for the failure class at this threshold  

This mimics how a maintenance team might balance “noise” (too many alerts) vs. catching as many real failures as possible.

---

## 🧱 Tech Stack

- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy` for data handling  
  - `scikit-learn` for modeling, pipelines, and GridSearchCV  
  - `matplotlib` / `seaborn` for plots  
  - `joblib` for saving the trained model  
- **Notebook:** Jupyter (`PredictiveMaintenance_INFD615.ipynb`)

---

## 🔮 Future Improvements

If I continue this project, I’d like to:

- Add time-series features (rolling averages, trends) instead of using each row independently
- Try gradient boosting (XGBoost/LightGBM) for potentially higher recall on rare failures
- Deploy a lightweight API or Streamlit dashboard to monitor live predictions
- Integrate cost-based evaluation (e.g., cost of downtime vs. cost of preventive maintenance)

---

**Thank you
