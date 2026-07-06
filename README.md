# House Price Valuation: Baseline Framework

A production-ready reference architecture for training an end-to-end baseline Generalized Linear Model (GLM). This framework standardizes data validation, feature engineering, and business-centric evaluation metrics for continuous target regression tasks.

---

## The Core Workflow

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
* **Qualitative Encoding & Domain Filtering:** 
  * Convert valid categorical text variables (e.g., `furnishingstatus`) into binary metrics using One-Hot Encoding (`pd.get_dummies(drop_first=True)`), bypassing the scaler for these 0/1 indicator flags.
  * Exclude high-cardinality columns with severe class imbalance 
    
### 3. Validation Strategy & Modeling
* **Leakage Prevention:** Split data into an 80% training set and a 20% unseen testing set *after* feature creation but *before* fitting models.
* **Algorithm:** Baseline Parametric Linear Regression (GLM).

---

## Baseline Reference Implementation

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
# get information about the dataframe

df.info()
df.describe()
df.isnull().sum().

# to replace columns with null values
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())
df['status_col'] = df['status_col'].fillna('Unknown')
df['target_col'] = df['target_col'].fillna(0)

# feature creation

df['bedroom_to_area_ratio'] = df['bedrooms'] / df['area']

# 1. Identify your columns explicitly by type
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'bedroom_to_area_ratio']
categorical_cols = ['mainroad', 'guestroom', 'furnishingstatus']

# 2. Extract your features from the dataframe
X_raw = df[numeric_cols + categorical_cols]
y = np.log1p(df['price'])

# 3. One-Hot Encode the categorical features
# drop_first=True prevents the "Dummy Variable Trap" (multicollinearity)
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=int)

# ----------------------------------------------------
# 2. Validation Split (Split FIRST to prevent leakage)
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# ----------------------------------------------------
# 3. Magnitude Alignment (Scale ONLY continuous numeric features)
# ----------------------------------------------------
scaler = StandardScaler()

# Fit and transform only on the training numeric columns
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# Transform only on the test numeric columns using training parameters
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ----------------------------------------------------
# 4. Model Training
# ----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------
# 5. Inverse Transformation & Evaluation
# ----------------------------------------------------
# Get raw predictions from the model (still in transformed space)
log_preds = model.predict(X_test)

# Reverse the transformation back to actual currency units 
# [NOTE: Swap out np.expm1() for np.sqrt() or remove entirely 
y_pred = np.expm1(log_preds)  
y_true = np.expm1(y_test)             

# Calculate Business Metric (MAE) in raw pounds
mae = mean_absolute_error(y_true, y_pred)
print(f"Model Baseline MAE: £{mae:,.2f}")
