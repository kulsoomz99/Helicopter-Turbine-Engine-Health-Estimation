# Helicopter-Turbine-Engine-Health-Estimation

## Methodology

### 1. Exploratory Data Analysis (EDA)
Initial analysis reveals that `trq_margin` (torque margin) is a strong predictor of fault status but has significant overlap between classes. This justifies a dedicated regression stage before classification.

### 2. Feature Engineering
We engineered 8 new features based on turbine thermodynamics:
*   `oat_mgt_diff`: Temperature rise across the engine.
*   `trq_per_ng`: Torque normalized by gas-generator speed.
*   `ng_minus_np`: Speed mismatch between gas generator and power turbine.
*   *Selection:* Features were selected using **Permutation Importance** and **Spearman Correlation** to remove noise and redundancy.

### 3. Stage 1: Probabilistic Regression
*   **Model:** XGBoost Regressor trained on Quantile Loss (10th, 50th, 90th percentiles).
*   **Distribution Fitting:** Residuals were analyzed against 4 distributions (Normal, Laplace, Student-T, Cauchy). The **Student-T distribution** was selected via BIC (Bayesian Information Criterion) to account for heavy tails in sensor noise.
*   **Output:** A predicted median (`loc`) and standard deviation (`scale`) for each observation.

### 4. Stage 2: Fault Classification
*   **Architecture:** LightGBM Classifier.
*   **Inputs:** Raw sensors + Engineered Features + **Stage 1 Outputs** (`pred_trq_margin`, `pred_std`).
*   **Why it works:** The classifier learns to trust the "expert opinion" of the regression model. If the regression predicts a highly negative torque margin with low uncertainty, the classifier confidently marks it as faulty.

### 5. Calibration & Scoring Optimization
The challenge score penalizes confident wrong predictions. We optimized this by:
1.  **Isotonic Calibration:** Using `CalibratedClassifierCV` to ensure predicted probabilities reflect true likelihoods.
2.  **Confidence Shrinkage:** Applying a power transformation to reduce confidence near the decision boundary, minimizing the penalty for borderline cases.

## Results

| Model | Metric | Score |
| :--- | :--- | :--- |
| **Baseline (Linear Reg)** | R² | ~0.973 |
| **Best Regressor (XGBoost)** | R² | **0.997** |
| **Best Classifier (LightGBM)** | F2-Score | **0.998** |
| **Calibrated Submission** | Challenge Score | **~0.997** |


## References

*   [PHM North America 2024 Data Challenge](https://phmsociety.org/)
*   [SHAP Library Documentation](https://shap.readthedocs.io/)


