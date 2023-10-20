# Understanding Logistic Regression in Machine Learning

## What is Logistic Regression?

Logistic regression is a fundamental and widely used statistical method in machine learning for binary classification problems based on one or more predictor variables. It's called "logistic" because it uses the logistic function (also known as the sigmoid function) to model the relationship between the predictors and the binary response. The logistic function ensures that the output is between 0 and 1, making it suitable for classification.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Assumptions in Logistic Regression

Logistic regression makes several key assumptions:
1. **Linearity:** The relationship between the predictors and the log-odds of the response variable is linear.
2. **Independence:** Observations should be independent of each other.
3. **No multicollinearity:** Predictor variables should not be highly correlated.
4. **Large sample size:** Logistic regression works best with a sufficiently large dataset.


## Why Not Linear Regression?

Linear regression is not suitable for binary classification tasks because it assumes a continuous output, while logistic regression produces probabilities between 0 and 1. Using linear regression for classification can lead to odd predictions 0>pred and pred>
1

## Difference Between Linear and Logistic Regression

Linear regression is used for predicting continuous values, while logistic regression is used for predicting binary categorical values. 

Linear regression assumes that error terms are normally distributed. In case of binary classification, this assumption does not hold true. For logistic regression the error terms follow the logistic distribution

Variance of Residual errors: Linear regression assumes that the variance of random errors is constant. This assumption is also violated in case of logistic regression.

## Log Loss

Log Loss, also known as cross-entropy loss, is a loss function used to evaluate the performance of logistic regression models. It measures the difference between predicted probabilities and actual class labels. Minimizing log loss leads a better model fit.

## Maximum Likelihood Estimation (MLE)

MLE is the method used to estimate the parameters of a statistical model. In logistic regression, MLE is used to find the parameter values that maximize the likelihood of the observed data.

## Pros and Cons of Logistic Regression

### Pros:
1. Simplicity and interpretability.
2. Works well for linearly separable data.
3. Provides probability estimates for classification.
4. Efficient with small to moderately sized datasets.

### Cons:
1. Assumes linearity, which may not hold in all cases.
2. May not perform well with highly complex data.
3. Prone to overfitting with a large number of predictors.

## Where is Logistic Regression Used?

Logistic regression finds applications in various fields, including:
- Medical diagnosis (e.g., disease prediction).
- Credit scoring (e.g., loan approval or denial).
- Marketing (e.g., customer churn prediction).