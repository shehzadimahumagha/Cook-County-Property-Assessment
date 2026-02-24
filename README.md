# Cook County Property Value Assessment

> Predicting residential property values for 10,000 Cook County homes using machine learning techniques in R.

---

## Project Overview

The **Cook County Assessor's Office (CCAO)** is responsible for determining the fair market value of over 1.8 million properties annually. Historically, valuation methods lacked transparency and precision, leading to public criticism. This project leverages historical property sales data and machine learning to improve the accuracy and fairness of property assessments.

**Objective:** Predict assessed values for 10,000 residential properties while minimizing Cross-Validated Mean Squared Error (CV MSE).

---

## Repository Structure

```
├── Final_code_CCAO.R            # Full R pipeline: data prep, modeling, predictions
├── assessed_values.csv          # Final predicted assessed values (pid + assessed_value)
├── Executive_Summary.pdf        # Full project write-up with methodology & results
└── README.md
```

> **Note:** The raw datasets (`historic_property_data.csv` and `predict_property_data.csv`) are not included due to file size. Contact the team or refer to the CCAO open data portal.

---

## Methodology

### 1. Data Preparation
- Selected 16 theoretically meaningful predictors of sale price
- Removed rows with missing values (`na.omit`)
- Encoded categorical variables using `model.matrix`
- Standardized continuous predictors for Lasso regression

### 2. Models Tested

| Model | CV MSE | R² |
|---|---|---|
| Linear Regression (baseline) | 15,952,188,508 | 83.46% |
| Lasso Regression | 15,914,348,180 | ~83.46% |
| **Forward Stepwise + Linear** | **15,883,616,317** | **83.46%** |
| Random Forest (10 trees) | 16,994,333,626 | 82.31% |

### 3. Final Model
**Linear Regression with Forward Stepwise Variable Selection** (via `stepAIC`) achieved the lowest CV MSE.

**Selected predictors:**
`meta_certified_est_bldg`, `meta_certified_est_land`, `econ_midincome`, `char_fbath`, `geo_black_perc`, `geo_tract_pop`, `char_bsmt`, `char_beds`, `econ_tax_rate`, `char_gar1_area`, `meta_class`, `geo_asian_perc`

---

## Results Summary

| Statistic | Value |
|---|---|
| Minimum | $0 |
| 1st Quartile | $109,002 |
| Median | $230,757 |
| Mean | $285,045 |
| 3rd Quartile | $359,631 |
| Maximum | $6,207,654 |

---

## How to Run

### Prerequisites
Install the following R packages:
```r
install.packages(c("caret", "dplyr", "glmnet", "MASS", "randomForest", "xgboost"))
```

### Steps
1. Clone this repository
2. Place `historic_property_data.csv` and `predict_property_data.csv` in the working directory
3. Open `Final_code_CCAO.R` in RStudio
4. Run the full script (~2–3 minutes runtime)
5. Output will be saved as `assessed_values.csv`

---

## Key Takeaways

- **Forward Stepwise Selection** outperformed Lasso and Random Forest on this dataset
- Certified building and land estimates (`meta_certified_est_bldg`, `meta_certified_est_land`) were the strongest predictors
- Random Forest underperformed, likely due to the limited number of trees (`ntree = 10`), increasing this could improve results
- The model supports CCAO's mission of **transparent, data-driven property assessments**

---

## License

This project was submitted as academic coursework for FIN 550. Please do not reproduce without permission.
