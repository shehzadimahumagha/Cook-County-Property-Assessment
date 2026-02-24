# PROJECT CODE SUBMISSION
# Names: Javeria Malik, Ahmed Bilal, Shehzadi Mahum Agha
#FULL RUN TIME - APPROXIMATELY 2-3 MINUTES

#importing the df
df <-historic_property_data <- read.csv("historic_property_data.csv")
df


# Install necessary libraries
#install.packages("xgboost")
#install.packages("caret")
#install.packages("dplyr")
library(xgboost)
library(caret)
library(dplyr)
#install.packages("glmnet")
library(glmnet)
#install.packages("MASS")


### LINEAR REGRESSION MODEL

# Select only the specified columns
df2 <- df %>% dplyr::select(
  sale_price,
  meta_certified_est_bldg,
  meta_certified_est_land,
  char_bldg_sf,
  meta_class,
  char_bsmt,
  char_gar1_area,
  econ_midincome,
  econ_tax_rate,
  char_fbath,
  char_rooms,
  char_beds,
  char_frpl,
  char_age,
  geo_black_perc,
  geo_asian_perc,
  geo_tract_pop
)


# View the resulting data frame
head(df2)

# Count missing values in each column
missing_values <- colSums(is.na(df2))

# Display the missing values for each column
missing_values


# Remove rows with missing values
df2 <- na.omit(df2)

# Create a linear model with sale_price as the dependent variable
lm_model <- lm(
  sale_price ~ 
    meta_certified_est_bldg +
    meta_certified_est_land +
    char_bldg_sf+
    char_age+
    char_rooms+
    meta_class+
    char_bsmt+
    char_gar1_area+
    econ_midincome+
    geo_tract_pop+
    geo_black_perc+
    geo_asian_perc+
    econ_tax_rate+
    char_fbath+
    char_beds+
    char_frpl,
  data = df2
)

# Summary of the model
summary(lm_model)

##CROSS VALIDATION

# Set the number of folds for cross-validation
cv_folds <- 5  # Adjust the number of folds as needed

# Set up training control for cross-validation
train_control <- trainControl(method = "cv", number = cv_folds, 
                              summaryFunction = defaultSummary, 
                              verboseIter = TRUE)

# Create the linear model formula (same as before)
formula <- sale_price ~ 
  meta_certified_est_bldg +
  meta_certified_est_land +
  char_bldg_sf+
  char_age+
  char_rooms+
  meta_class+
  char_bsmt+
  char_gar1_area+
  econ_midincome+
  geo_tract_pop+
  geo_black_perc+
  geo_asian_perc+
  econ_tax_rate+
  char_fbath+
  char_beds+
  char_frpl

# Perform k-fold cross-validation using the lm model
cv_model <- train(formula, data = df2, method = "lm", trControl = train_control)

# Display the cross-validation results
cv_results <- cv_model$results
print(cv_results)

# Get the RMSE (Root Mean Squared Error) and calculate MSE (Mean Squared Error)
cat("Cross-validated RMSE:", cv_results$RMSE, "\n")
cat("Cross-validated MSE:", cv_results$RMSE^2, "\n")


####PERFORMING LASSO ON LINEAR MODEL


# Prepare the data
# Exclude the target variable from the predictors
X <- df2[, setdiff(names(df2), "sale_price")]
y <- df2$sale_price

# Convert categorical variables to dummy variables (use model.matrix)
X_dummy <- model.matrix(~ . - 1, data = X)  # '-1' to remove intercept

# Standardize the predictor variables (important for Lasso)
X_scaled <- scale(X_dummy)

# Check which columns have missing values
col_missing <- colSums(is.na(X))
col_missing[col_missing > 0]

# Fit the Lasso model using glmnet (alpha = 1 for Lasso)
lasso_model <- glmnet(X_scaled, y, alpha = 1)

# Plot the Lasso path (coefficients as a function of lambda)
plot(lasso_model, xvar = "lambda", label = TRUE)

# Cross-validation to find the best lambda (penalty parameter)
cv_lasso <- cv.glmnet(X_scaled, y, alpha = 1)

# Best lambda from cross-validation
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Fit the final model with the best lambda
final_lasso_model <- glmnet(X_scaled, y, alpha = 1, lambda = best_lambda)

# Summary of cross-validation results
plot(cv_lasso)

# MSE at the best lambda
best_mse <- min(cv_lasso$cvm)
cat("MSE at best lambda:", best_mse, "\n")


###Identifying important variables in Lasso model
# Get the coefficients from the final Lasso model
lasso_coefs <- coef(final_lasso_model)
lasso_coefs
# Identify the variables with non-zero coefficients (excluding the intercept)
important_vars <- rownames(lasso_coefs)[lasso_coefs[, 1] != 0 & rownames(lasso_coefs) != "(Intercept)"]

# Check that the important variables are present in the original dataframe
# Ensure the variable names in 'important_vars' exist in the dataframe's columns
important_vars <- intersect(important_vars, colnames(df))

# Subset the dataset to include only the important variables
df3 <- df2[, important_vars]

# Print the important variables
cat("Important variables:\n")
print(important_vars)


# Create the linear model using only the important variables from df2
lm_model2 <- lm(formula,
                data = df2)

# Summary of the model
summary(lm_model2)

#K-FOLD TRAINING TO CALCULATE MSE for LASSO
# Load necessary libraries
library(caret)

# Set the number of folds for cross-validation
cv_folds2 <- 5  # You can adjust the number of folds as needed

# Set up training control
train_control2 <- trainControl(method = "cv", number = cv_folds2, 
                               summaryFunction = defaultSummary)

# Train the model with cross-validation
cv_model2 <- train(formula, 
                   data = df2, method = "lm", trControl = train_control2)

# Get the cross-validation results
cv_results2 <- cv_model2$results
print(cv_results2)

# Extract MSE (Mean Squared Error) from the cross-validation results
cat("Cross-validated MSE:", cv_results2$RMSE^2, "\n")
cat("Cross-validated RMSE:", cv_results2$RMSE, "\n")


###PERFORMING FORWARD STEPWISE MODEL ON LINEAR MODEL


# Load necessary library
library(MASS)

full_model <- lm(
  formula, 
  data = df2
)

# Define the null model (only includes the intercept)
null_model <- lm(sale_price ~ 1, data = df2)

# Perform forward stepwise selection
stepwise_model <- stepAIC(null_model, 
                          scope = list(lower = null_model, upper = full_model), 
                          direction = "forward", 
                          trace = TRUE)

# Display the summary of the final model
summary(stepwise_model)

# Compare the final model's MSE with the previous model
final_rmse <- sqrt(mean(stepwise_model$residuals^2))
cat("Final Model RMSE:", final_rmse, "\n")
cat("Final Model MSE:", final_rmse^2, "\n")
#WE SEE THAT FORWARD STEPWISE MODEL HAS THE BEST MSE OUT OF ALL OUR MODELS


# Extract the formula from the stepwise model
best_model_formula <- stepwise_model$call[[2]]
best_model_formula

#Setting up the final model 
final_model<-lm(formula=best_model_formula, data=df2)
summary(final_model)


#RANDOM FOREST
library(randomForest)
# Define the formula for the model
formula_rf <- sale_price ~
  meta_certified_est_bldg +
  meta_certified_est_land +
  char_bldg_sf+
  char_age+
  char_rooms+
  meta_class+
  char_bsmt+
  char_gar1_area+
  econ_midincome+
  geo_tract_pop+
  geo_black_perc+
  geo_asian_perc+
  econ_tax_rate+
  char_fbath+
  char_beds+
  char_frpl



# Train the Random Forest model
set.seed(123) # For reproducibility
rf_model <- randomForest(
  formula = formula_rf,
  data = df2,
  ntree = 10,      # Number of trees
  mtry = 5,         # Number of predictors randomly sampled at each split
  importance = TRUE # Calculate variable importance
)

# View the Random Forest model summary
print(rf_model)

# View the variable importance
importance(rf_model)



##PREDICTIONS
library(dplyr)

# Load the test dataset
predict_property_data <- read.csv('predict_property_data.csv')
# Check if predictions are being generated correctly
predict_property_data$assessed_value <- predict(final_model, newdata = predict_property_data)

# Ensure that the predicted values are non-NA and non-negative
predict_property_data <- predict_property_data %>%
  mutate(assessed_value = ifelse(is.na(assessed_value) | assessed_value < 0, 0, assessed_value))

# Check the first few rows after prediction
head(predict_property_data)

output <- predict_property_data %>%
  dplyr::select(pid, assessed_value)

head(output)

# Export output_file to CSV
write.csv(output, "assessed_values.csv", row.names = FALSE)

# Summary statistics
# Convert to numeric if needed
output$assessed_value <- as.numeric(output$assessed_value)

summary_stats <- summary(output$assessed_value)
print(summary_stats)


