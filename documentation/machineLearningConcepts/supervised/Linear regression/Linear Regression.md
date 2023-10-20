# Exploring Linear Regression in Python


## 1. What is a Linear Regression?

Linear regression is a statistical method used for modeling the linear relationship between a dependent variable Y and one or more independent variables x1,x2,x3,...,xn by fitting a linear equation to the observed data. It can also be described as fitting a hyperplain(line in 1D)

It should be noted that linear is very sensitive to outliers

                            Y=a + b1x1 + b2x2 +...+ bmxm + e

```python
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()
model.fit(x_values, y_values)
```

## 2. Critical Assumptions of Linear Regression

Linear regression relies on several key assumptions, including linearity, independence, homoscedasticity, and normality.

You can test these assumptions in Python using various statistical tests and visualizations.

Linearity: It is imperative to have a linear relationship between the dependent and independent variables. A scatter plot can validate this

Independence: Multicollinearity should not occur between the independent variables in the dataset. That is the independent variables should not be correlated

Homoscedasticity: The variance of the residuals (the error terms) is the same across all values of the independent variables

Normality: The residuals (the error terms) are independent and Normally distributed

## 3. What are the consequences of Heteroscedasticity?

Heteroscedasticity is a violation of the assumption that the variance of the errors is constant(Homoscedasticity). 

It causes the coefficient estimates to be less preccise(Not reflect the population)

Heteroscedasticity tends to produce p-values that are smaller than they should be. This problem can lead you to conclude that a model term is statistically significant when it is actually not

- https://statisticsbyjim.com/regression/heteroscedasticity-regression/


You can detect heteroscedasticity in Python by plotting the residuals or using statistical tests. Here's a code snippet for visualization:

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Check for heteroscedasticity
residuals = model.resid
plt.scatter(model.predict(), residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values")
```

## 4. Difference Between R-squared and Adjusted R-squared

R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables. R-squared tends to increase as the number of independent variables increases.

Adjusted R-squared penalizes for adding unnecessary independent variables. 


```python
from sklearn.metrics import r2_score

# Calculate R-squared
r_squared = r2_score(y_true, y_pred)

# Calculate Adjusted R-squared
n = len(y)
p = len(features)
adjusted_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))
```

## 5. Improving Linear Regression Accuracy

There are multiple ways to increase the accuracy of a linear regression model
Feature Selection
Feature Engineering
Regularization
Removing Outliers

## 6. Interpreting a Q-Q Plot

A Q-Q plot (Quantile-Quantile plot) is used to assess if a dataset follows a particular theoretical distribution. In linear regression, it can help you check the assumption of normality of residuals. You can create a Q-Q plot in Python with:

```python
import statsmodels.api as sm

# Create a Q-Q plot
sm.qqplot(residuals, line='r')
plt.show()
```

## 7. Importance of the F-Test

 
The F-test in the context of linear regression is a statistical test used to determine whether a linear regression model with multiple independent variables is a better fit for the data than a model with no independent variables (i.e., a model with only an intercept term)

In common vernacular, an f-test basically checks 'is my model any good or should I just chuck it out the window'?

In linear regression, we typically have two models to compare:

- The Full Model: This is the model with multiple independent variables (features).

- The Reduced Model: This is the model with only an intercept term (no independent variables).

The F-test assesses whether the full model significantly improves the fit compared to the reduced model. In other words, it evaluates whether including independent variables in the model leads to better predictions.

The null and alternative hypotheses for the F-test are:

Null Hypothesis (H0): The full model is not significantly better than the reduced model. In mathematical terms, it means that all the coefficients of the independent variables in the full model are zero.

Alternative Hypothesis (H1): The full model is significantly better than the reduced model, i.e., at least one of the coefficients in the full model is not equal to zero.

If the F-statistic (the test statistic) is significantly larger than what we would expect by chance, we reject the null hypothesis, indicating that the full model is a better fit.

- https://online.stat.psu.edu/stat462/node/135/


```python
import statsmodels.api as sm

# Fit the full model with independent variables
X = sm.add_constant(independent_variables)  # Add a constant (intercept)
model_full = sm.OLS(dependent_variable, X).fit()

# Fit the reduced model (intercept only)
model_reduced = sm.OLS(dependent_variable, [1] * len(dependent_variable)).fit()

# Perform the F-test
f_statistic = ((model_reduced.ssr - model_full.ssr) / (model_full.df_model - model_reduced.df_model)) / (model_full.ssr / model_full.df_resid)
p_value = 1 - model_full.f_test("const = 0").pvalue

# Print the F-statistic and p-value
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Check if the result is statistically significant
alpha = 0.05  # Significance level
if p_value < alpha:
    print("The full model is statistically better than the reduced model.")
else:
    print("There is no significant evidence that the full model is better than the reduced model.")
```

## 8. Disadvantages of Linear Regression

Linear regression assumes a linear relationship, which may not always be true. It's also sensitive to outliers, and violations of its assumptions can lead to inaccurate results.

## 9. Multicollinearity

Multicollinearity occurs when independent variables in a regression model are highly correlated making it challenging to separate their individual effects on the dependent variable

### Testing for Multicollinearity:
1. Correlation Matrix:
Calculate the correlation matrix for the independent variables. If you observe high correlations (typically above 0.7 or -0.7), it's an indication of multicollinearity.

2. Variance Inflation Factor (VIF):
Variance inflation factor measures how much the behavior (variance) of an independent variable is influenced, or inflated, by its interaction/correlation with the other independent variables. Variance inflation factors allow a quick measure of how much a variable is contributing to the standard error in the regression.

from statsmodels.stats.outliers_influence import variance_inflation_factor

```python
# Calculate VIF for each variable
vif = pd.DataFrame()
vif["Variable"] = your_data.columns
vif["VIF"] = [variance_inflation_factor(your_data.values, i) for i in range(your_data.shape[1])]

# Display the VIF values
print(vif)
```


- https://www.investopedia.com/terms/v/variance-inflation-factor.asp#:~:text=Variance%20inflation%20factor%20measures%20how,standard%20error%20in%20the%20regression.


## 10. Fixes for Multicollinearity

1. **Remove One of the Correlated Variables**:
   If two or more variables are highly correlated, consider removing one of them. This can help eliminate multicollinearity, but it may lead to a loss of information.

2. **Combine Variables**:
   In some cases, you can create new variables by combining correlated ones. This can reduce multicollinearity while retaining the information.

3. **Principal Component Analysis (PCA)**:
   PCA is a dimensionality reduction technique that can transform correlated variables into a set of uncorrelated variables (principal components). You can use PCA to create new variables that are orthogonal to each other.

   ```python
   from sklearn.decomposition import PCA

   # Create a PCA model
   pca = PCA(n_components=2)  # Choose the number of components
   pca.fit(your_data)

   # Transform the data using the first two principal components
   transformed_data = pca.transform(your_data)
   ```

4. **Regularization**:
   Ridge and Lasso regression introduce penalty terms that can reduce the magnitude of coefficients and, in turn, mitigate multicollinearity.

   ```python
   from sklearn.linear_model import Ridge, Lasso

   # Create Ridge or Lasso regression models
   ridge_model = Ridge(alpha=1.0)  # Adjust alpha as needed
   lasso_model = Lasso(alpha=1.0)  # Adjust alpha as needed

   # Fit the models
   ridge_model.fit(X, y)
   lasso_model.fit(X, y)
   ```

5. **Feature Selection**:
   Use feature selection techniques to choose the most relevant variables and exclude the redundant ones. This can help in reducing multicollinearity.

   ```python
   from sklearn.feature_selection import SelectKBest, f_regression

   # Create a SelectKBest feature selector
   selector = SelectKBest(score_func=f_regression, k=3)  # Choose 'k' features
   selector.fit(X, y)

   # Get selected features
   selected_features = X.columns[selector.get_support()]
   ```

When dealing with multicollinearity, it's essential to carefully consider the nature of your data and the goals of your analysis to choose the most appropriate method for addressing the issue. Different approaches may be more suitable in various situations.


## 11. Regression vs. Classification
Regression and classification are categorized under the same umbrella of supervised machine learning.
Regression predicts a continuous output, while classification predicts a discrete output. 

## 12. Finding the Optimal Linear Regression Line

By minimizing the sum of squared residuals

## 13. Performance Metrics for Linear Regression

To estimate the efficiency of a linear regression model, you can use metrics like 
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE),
- Root Mean Squared Error (RMSE),
- R-squared

Heteroskedasticity is a statistical phenomenon in linear regression where the variance of the error terms is not constant across all levels of the independent variables. In simpler terms, it means that the spread of the residuals (the differences between the observed and predicted values) changes as you move along the independent variable(s). Detecting heteroskedasticity is crucial as it can lead to incorrect conclusions about the statistical significance of regression coefficients and affect the reliability of your model.

To test for heteroskedasticity, you can use visual methods and formal statistical tests. Here's a step-by-step guide:

### Visual Methods

1. **Residual Plots**: Plot the residuals against the predicted values. If you see a funnel-shaped pattern, where the spread of residuals increases or decreases systematically, it's an indication of heteroskedasticity.

   ```python
   import matplotlib.pyplot as plt

   # Calculate residuals (res) from your linear regression model
   res = model.resid

   # Plot residuals against predicted values
   plt.scatter(model.predict(), res)
   plt.xlabel("Predicted Values")
   plt.ylabel("Residuals")
   plt.title("Residuals vs. Predicted Values")
   plt.show()
   ```

   If the spread of residuals widens or narrows as you move along the x-axis (predicted values), this is a sign of heteroskedasticity.

2. **Histogram of Residuals**: Plot a histogram of the residuals. If the distribution is not approximately symmetrical or looks skewed, it might indicate heteroskedasticity.

   ```python
   plt.hist(res, bins=20)
   plt.title("Histogram of Residuals")
   plt.show()
   ```

   If you observe irregular shapes in the histogram, it could be a sign of heteroskedasticity.

### Formal Statistical Tests

1. **Breusch-Pagan Test**: This test is used to formally check for heteroskedasticity. The null hypothesis is that there is homoskedasticity (constant variance), and the alternative hypothesis is that there is heteroskedasticity.

   ```python
   from statsmodels.stats.diagnostic import het_breuschpagan

   # Perform the Breusch-Pagan test
   _, p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)

   # Check the p-value
   if p_value < 0.05:
       print("Heteroskedasticity is detected.")
   else:
       print("No evidence of heteroskedasticity.")
   ```

   If the p-value is less than your chosen significance level (usually 0.05), you reject the null hypothesis, indicating the presence of heteroskedasticity.

2. **White's Test**: White's test is another formal test for heteroskedasticity. It's similar to the Breusch-Pagan test but uses a different approach.

   ```python
   from statsmodels.stats.diagnostic import het_white

   # Perform White's test
   _, p_value, _, _ = het_white(model.resid, model.model.exog)

   # Check the p-value
   if p_value < 0.05:
       print("Heteroskedasticity is detected.")
   else:
       print("No evidence of heteroskedasticity.")
   ```

Keep in mind that heteroskedasticity can impact the reliability of your regression results. If you detect heteroskedasticity, consider using techniques like weighted least squares (WLS) regression, which can be more appropriate for heteroskedastic data, or transforming your variables to stabilize the variance.