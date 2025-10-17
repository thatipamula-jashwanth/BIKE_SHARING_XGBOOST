# Bike Sharing Demand Prediction with XGBoost

## Project Overview
This project predicts **hourly bike rental demand** in Seoul using the **Seoul Bike Sharing dataset**.  
We use **XGBoost regression**, a gradient boosting algorithm that builds sequential decision trees where each tree corrects the errors of previous trees.

---

## Dataset
- **Source:** Seoul Bike Sharing Dataset  
- **Rows:** 8,760 (hourly data for one year)  
- **Columns:** 14  
  - Features: Hour, Temperature, Humidity, Wind speed, Visibility, Dew point temperature, Solar Radiation, Rainfall, Snowfall, Seasons, Holiday, Functioning Day, Date  
  - Target: Rented Bike Count  

- **Categorical columns:** Date, Seasons, Holiday, Functioning Day  

---

## XGBoost Overview
**XGBoost** is a gradient boosting decision tree algorithm that:

1. Builds trees sequentially.  
2. Each tree predicts the residuals (errors) of the previous trees.  
3. Final prediction = sum of predictions from all trees (weighted by learning rate).  

**Advantages:**  
- Handles non-linear relationships well  
- Robust to outliers  
- Regularization to reduce overfitting  
- Efficient and fast for medium-sized datasets  

**Key Hyperparameters:**
- `n_estimators`: number of trees  
- `max_depth`: maximum tree depth  
- `learning_rate`: step size for each tree (shrinkage)  
- `subsample`: fraction of rows per tree  
- `colsample_bytree`: fraction of features per tree  
- `reg_alpha`, `reg_lambda`: L1 and L2 regularization  

---

## Preprocessing Steps
1. **Load Dataset** with proper encoding (Latin-1 for special characters).  
2. **Identify categorical columns** automatically.  
3. **One-hot encoding** for categorical features.  
4. **Split dataset** into training (80%) and validation (20%) sets.  
5. **Log-transform target variable** (`y = log1p(y)`) to stabilize variance and handle zeros.  

---

## Hyperparameter Tuning
- **Original model**: high overfitting  
  - Train R² = 0.9984, Validation R² = 0.9468  
  - RMSE train ≈ 25.4, RMSE val ≈ 148.9 → overfit  

- **Tuned model changes:**  
  - Decreased `max_depth` → reduces overfitting  
  - Increased `n_estimators` → allows more trees to learn  
  - Increased `subsample` → row sampling to reduce overfitting  
  - Increased `reg_alpha` → L1 regularization on leaves  
  - Learning rate adjusted  

- **Tuned model performance:**  

| Metric | Training | Validation |
|--------|----------|------------|
| R²     | 0.9654   | 0.9420     |
| RMSE   | 119.92   | 155.39     |
| MAE    | 68.13    | 84.21      |

 Values are now close between train and validation → minimal overfitting.
