# Understanding Ridge and Lasso Regression in Python

## What are Ridge and Lasso Regression?
Ridge and Lasso Regression are types of linear regression models using regularization to help solve overfitting. Regularization prevents overfitting by adding a penalty term to the cost function

The main difference lies in the regularization term:

- Ridge Regression uses L2 regularization, which adds the sum of squared coefficients to the cost function.
- Lasso Regression uses L1 regularization, which adds the sum of absolute values of coefficients to the cost function.

Ridge tends to shrink coefficients towards zero and helps with multicollinearity, while Lasso can lead to exact zero coefficients, effectively performing feature selection.

## How do Ridge and Lasso Regression algorithms work?

### Ridge Regression:
Ridge Regression, also known as L2 regularization, adds the sum of the squares of the coefficients to the cost function. This penalizes large coefficients

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
```

### Lasso Regression:
Lasso Regression, or L1 regularization, adds the sum of the absolute values of the coefficients to the cost function. It encourages the model to perform feature selection by setting some coefficients to zero.

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)
```

## What is the difference between ridge and lasso regression?
The main difference lies in the regularization term:

- Ridge Regression uses L2 regularization, which adds the sum of squared coefficients to the cost function.
- Lasso Regression uses L1 regularization, which adds the sum of absolute values of coefficients to the cost function.

Ridge tends to shrink coefficients towards zero, while Lasso can lead to exact zero coefficients, effectively performing feature selection.


## Can you list out the critical assumptions of Ridge and Lasso Regression?

Ridge and Lasso Regression assume a linear relationship between the features and the target variable, and they assume that the error terms are normally distributed and have constant variance.
**assumptions**:
- Linear relationship between the features and the target variable
- Error terms are normally distributed and have constant variance

## What are the pros and cons of Ridge and Lasso Regression?

### Pros:
- Regularization helps prevent overfitting.
- Ridge and Lasso can handle multicollinearity.
- Lasso performs feature selection.

### Cons:
- They may not perform well when the true relationship between features and target is highly non-linear.
- The choice of the regularization strength (alpha) can be challenging.
- Interpretability may be reduced due to coefficient shrinkage.

## Which is better at reducing the variance between test and train datasets: Ridge or Lasso Regression?
Lasso Regression is generally better at reducing the variance between test and train datasets. This is because it removes variables increasing but reducing the variance

## What scenario would Ridge Regression be better than Lasso Regression?
Ridge Regression is better when you suspect that all features are relevant.


## What does lambda do in Ridge Regression?
Lambda, represented as alpha in scikit-learn, is a hyperparameter that controls the strength of regularization in Ridge Regression. A higher alpha value results in stronger regularization

## What role does cross-validation play in Ridge Regression?
Cross-validation is used to select the best alpha value in Ridge Regression.

## Can Ridge Regression work with 100 data points and 100 features?
No the data points should always be greater than number of features
