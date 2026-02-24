# Cook County Property Value Assessment

> Predicting residential property values for 10,000 Cook County homes using machine learning techniques in R.

---

## Project Overview

The **Cook County Assessor's Office (CCAO)** is responsible for determining the fair market value of over 1.8 million properties annually, a process critical for property tax revenue generation. Historically, valuation methods lacked transparency and precision, leading to public criticism. To address these challenges, CCAO embraced data science and machine learning techniques to enhance valuation accuracy.

As a data scientist on the team, the objective was to predict the residential property values of 10,000 homes as accurately as possible. This involved leveraging historical property sales data to develop models that minimize the Cross-Validated Mean Squared Error (CV MSE) of predictions, providing actionable insights and improvements for property assessments.

---

## Repository Structure

```
├── Final_code_CCAO.R            # Full R pipeline: data prep, modeling, predictions
├── assessed_values.csv          # Final predicted assessed values (pid + assessed_value)
├── Executive_Summary.pdf        # Full project write-up with methodology & results
└── README.md
```

> **Note:** The raw datasets (`historic_property_data.csv` and `predict_property_data.csv`) are not included due to file size.

---

## Methodology

### 1. Data Preparation

The first step involved cleaning and preparing the data. This included:

- Handling missing values via `na.omit()`
- Encoding categorical variables using `model.matrix`
- Standardizing continuous predictors (required for Lasso regression)
- Dropping variables with excessive missing values to ensure model consistency

The following 16 predictors were selected based on logical relevance to property sale prices:

| Variable | Description |
|---|---|
| `meta_certified_est_bldg` | Certified building estimate |
| `meta_certified_est_land` | Certified land estimate |
| `char_bldg_sf` | Building square footage |
| `char_age` | Age of property |
| `char_rooms` | Number of rooms |
| `char_fbath` | Number of full bathrooms |
| `char_beds` | Number of bedrooms |
| `char_bsmt` | Basement type |
| `char_gar1_area` | Garage area |
| `char_frpl` | Number of fireplaces |
| `meta_class` | Property class |
| `econ_midincome` | Median income in area |
| `econ_tax_rate` | Local tax rate |
| `geo_tract_pop` | Census tract population |
| `geo_black_perc` | % Black population in tract |
| `geo_asian_perc` | % Asian population in tract |

---

### 2. Model Development

Four modeling approaches were tested, each evaluated using 5-fold cross-validation.

---

#### Model 1 — Linear Regression (Baseline)

A standard OLS linear regression was fit using all 16 predictors as a baseline. 5-fold cross-validation was used to evaluate generalization performance.

**Results:**
- CV RMSE: 126,302
- CV MSE: 15,952,188,508
- R²: 83.46%

**Notable coefficient estimates:**

| Variable | Estimate | Significance |
|---|---|---|
| `meta_certified_est_bldg` | 0.9322 | *** |
| `meta_certified_est_land` | 1.578 | *** |
| `char_bsmt` | -7,801 | *** |
| `char_gar1_area` | 8,823 | *** |
| `econ_midincome` | 0.5089 | *** |
| `geo_tract_pop` | -4.993 | *** |
| `geo_black_perc` | -45,180 | *** |
| `char_fbath` | 21,790 | *** |
| `char_beds` | -5,668 | *** |
| `char_bldg_sf` | 1.055 | Not significant |
| `char_age` | 19.20 | Not significant |
| `char_frpl` | 168.0 | Not significant |

---

#### Model 2 — Lasso Regression

Lasso regression was applied to penalize less important predictors and shrink their coefficients to zero. The optimal penalty parameter lambda was selected via cross-validation.

- Best lambda: **449.5587**
- CV RMSE: 126,152.1
- CV MSE: **15,914,348,180**
- R²: ~83.46%

Lasso identified the following variables as unimportant (coefficients shrunk to zero):
`char_age`, `char_rooms`, `char_bsmt`, `geo_black_perc`

The remaining important variables retained by Lasso:
`meta_certified_est_bldg`, `meta_certified_est_land`, `meta_class`, `char_bsmt`, `char_gar1_area`, `econ_midincome`, `econ_tax_rate`, `char_fbath`, `char_beds`, `geo_black_perc`, `geo_asian_perc`, `geo_tract_pop`

The CV MSE from Lasso was lower than the baseline linear model, confirming it is a better fit given the data structure.

---

#### Model 3 — Forward Stepwise Selection + Linear Regression (Final Model)

Forward stepwise selection was implemented using `stepAIC()` from the `MASS` package. Starting from a null model (intercept only), variables were added one at a time based on their statistical significance and AIC improvement.

**Selected predictors:**

```
sale_price ~ meta_certified_est_bldg + meta_certified_est_land +
             econ_midincome + char_fbath + geo_black_perc + geo_tract_pop +
             char_bsmt + char_beds + econ_tax_rate + char_gar1_area +
             meta_class + geo_asian_perc
```

**Results:**
- Final Model RMSE: 126,030.2
- Final Model MSE: **15,883,616,317** (lowest among all models)
- R²: 83.46%
- Residual Standard Error: 126,000 on 42,697 degrees of freedom

**Coefficient estimates from final model:**

| Variable | Estimate | Std. Error | t-value | Significance |
|---|---|---|---|---|
| (Intercept) | -46,080 | 8,764 | -5.258 | *** |
| `meta_certified_est_bldg` | 0.9329 | 0.004833 | 193.035 | *** |
| `meta_certified_est_land` | 1.582 | 0.01826 | 86.601 | *** |
| `econ_midincome` | 0.5073 | 0.02359 | 21.508 | *** |
| `char_fbath` | 22,320 | 1,160 | 19.246 | *** |
| `geo_black_perc` | -44,960 | 2,335 | -19.250 | *** |
| `geo_tract_pop` | -5.004 | 0.3760 | -13.310 | *** |
| `char_bsmt` | -7,897 | 639.5 | -12.350 | *** |
| `char_beds` | -4,853 | 665.7 | -7.290 | *** |
| `econ_tax_rate` | 410.3 | 147.2 | 2.788 | ** |
| `char_gar1_area` | 8,588 | 2,302 | 3.731 | *** |
| `meta_class` | 103.0 | 24.94 | 4.131 | *** |
| `geo_asian_perc` | -21,050 | 8,320 | -2.530 | * |

All selected predictors are statistically significant at the 5% level or better.

---

#### Model 4 — Random Forest

A Random Forest ensemble model was trained as a benchmark using the same 16 initial predictors.

**Configuration:**
- Number of trees: 10
- Variables tried at each split (mtry): 5
- Importance: TRUE

**Results:**
- Mean of Squared Residuals: **16,994,333,626**
- % Variance Explained: 82.31%

The Random Forest CV MSE was higher than all linear models, likely due to the limited number of trees (`ntree = 10`). Increasing `ntree` could potentially improve performance.

---

### 3. Model Comparison Summary

| Model | CV RMSE | CV MSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 126,302 | 15,952,188,508 | 83.46% |
| Lasso Regression | 126,152 | 15,914,348,180 | 83.46% |
| **Forward Stepwise + Linear** | **126,030** | **15,883,616,317** | **83.46%** |
| Random Forest (10 trees) | — | 16,994,333,626 | 82.31% |

The **Forward Stepwise Linear Regression** model was selected as the final model due to its lowest CV MSE, interpretability, and parsimony.

---

## Predictions & Results

The final model was applied to `predict_property_data.csv` to generate assessed values for 10,000 properties. Negative or missing predictions were replaced with 0.

### Summary Statistics of Predicted Assessed Values

| Statistic | Value |
|---|---|
| Minimum | $0 |
| 1st Quartile | $109,002 |
| Median | $230,757 |
| Mean | $285,045 |
| 3rd Quartile | $359,631 |
| Maximum | $6,207,654 |

The mean ($285,045) being notably higher than the median ($230,757) indicates a right-skewed distribution, consistent with typical real estate markets. The high maximum value suggests the presence of extreme outliers or properties with exceptional characteristics.

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

- Forward Stepwise Selection produced the most accurate and parsimonious model, outperforming Lasso and Random Forest on this dataset.
- Certified building and land estimates (`meta_certified_est_bldg`, `meta_certified_est_land`) were the strongest predictors of sale price, with t-values of 193 and 87 respectively.
- Socioeconomic and geographic variables (`econ_midincome`, `geo_black_perc`, `geo_tract_pop`) were statistically significant, reflecting the role of neighborhood characteristics in property valuation.
- Random Forest underperformed due to the constrained number of trees; this is worth revisiting with a larger `ntree` value.
- The model supports CCAO's mission of transparent, data-driven, and fair property assessments by reducing reliance on opaque manual valuation methods.
