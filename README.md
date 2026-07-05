# House Price Valuation: Baseline Framework

A production-ready reference architecture for training an end-to-end baseline Generalized Linear Model (GLM). This framework standardizes data validation, feature engineering, and business-centric evaluation metrics for continuous target regression tasks.

---

## 🚀 The Core Workflow Blueprint

### 1. Data Identification & Discovery
* **Evaluation Matrix:** Check `df.info()`, `df.describe()`, and `df.isnull().sum()`.
* **Distribution Assessment:** Evaluate the `min`, `50%` (median), and `max` values to identify target distribution shapes.

### 2. Preprocessing & Feature Engineering
* **Feature Creation:** `bedroom_to_area_ratio` (captures layout density).
* **Feature Scale:** Retain raw `area` and `bedrooms` to preserve physical scale.
* **Magnitude Alignment:** Apply `StandardScaler` to input features ($X$) to prevent large-scale dimensions (e.g., square footage) from mathematically dominating smaller dimensions (e.g., room counts).
* **Target Treatment:** 
  * *Right-Skewed (Huge Max):* Use `np.log1p()` to compress the tail and satisfy normal error assumptions.
  * *Left-Skewed (Tiny Min):* Use `np.square()` to expand the tail.
  * *Symmetric/Normal:* Leave target raw.

### 3. Validation Strategy & Modeling
* **Leakage Prevention:** Split data into an 80% training set and a 20% unseen testing set *after* feature creation but *before* fitting models.
* **Algorithm:** Baseline Parametric Linear Regression (GLM).

---

## 💻 Baseline Reference Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ----------------------------------------------------
# 1. Feature Engineering & Target Separation
# ----------------------------------------------------
df['bedroom_to_area_ratio'] = df['bedrooms'] / df['area']

X = df.drop(columns=['price'])
y = np.log1p(df['price']) # Handling right-skewed target

# ----------------------------------------------------
# 2. Scaling & Validation Split
# ----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 3. Model Training
# ----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------
# 4. Inverse Transformation & Evaluation
# ----------------------------------------------------
log_preds = model.predict(X_test)

# CRITICAL: Reverse log space back to actual currency units
real_preds = np.expm1(log_preds)
real_actuals = np.expm1(y_test)

# Business Metric: Mean Absolute Error (MAE)
mae = mean_absolute_error(real_actuals, real_preds)
print(f"Model Baseline MAE: £{mae:,.2f}")
