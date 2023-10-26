# Demystifying Decision Trees (CART) in Machine Learning

Decision Trees, particularly the CART  algorithm, are a fundamental tool in the machine learning toolbox. Let's explore these questions and provide Python code for each.

## 1. What is Decision Trees (CART)?

Decision Trees are a machine learning algorithm that builds a tree-like structure to make predictions. In the case of CART (Classification and Regression Trees), they can handle both classification and regression tasks.
Decision Trees work by recursively splitting the dataset based on features to create a tree structure minimizing impurity or maximizing information gain at each split.
Assumptions:
- Independence of features

```python
# Example of creating a Decision Tree Classifier in Python using scikit-learn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
clf = DecisionTreeClassifier()
clf = DecisionTreeRegressor()
```

## 3. What are the main use cases for Decision Trees (CART) in machine learning?

Decision Trees are versatile and can be used for both classification and regression tasks. Common use cases include medical diagnosis, fraud detection, and predicting housing prices.
They are useful when the data is non-linear, and interpretability is essential but They may be less reliable for high-dimensional data or data with complex relationships.


## 6. What does “impurity” refer to?

Impurity in Decision Trees refers to the disorder or randomness of data within a node and defines the quality of the split. 

The Gini impurity and entropy are common measures of impurity.

**Gini Impurity**: Gini impurity measures the probability of incorrectly classifying a randomly chosen element from the dataset if it were labeled randomly according to the class distribution in that node. It is calculated as the sum of the squared probabilities of each class being chosen incorrectly.

**Entropy**: Entropy is a measure of the level of disorder or uncertainty in a dataset. In the context of Decision Trees, it quantifies the uncertainty of the class distribution in a node. High entropy means that the classes are more evenly distributed and, therefore, more disordered, while low entropy indicates a more organized and homogenous distribution.

## 7. What does “information gain” refer to?

Information gain measures how much new information or knowledge is acquired when a dataset is split based on a particular feature. It quantifies the reduction in impurity.

## 10. What does it mean that decision trees are "greedy"?

Decision Trees are considered "greedy" because they make optimal decisions at each node without considering subsequent splits. Greedy algorithms aren’t guaranteed to find the global optimum

## 11. How does Decision Trees handle both classification and regression tasks, and what are the differences between these applications?

For classification, Decision Trees assign the majority class label to a leaf node. In regression, they assign the average target value. The key difference is in what they predict.

## 12. How are decision trees prone to bias?

Decision Trees can be biased towards the majority class when dealing with imbalanced datasets. They tend to favor the majority class, leading to misclassifications in minority classes.


## 14. What are the pros and cons of decision trees?

**Pros:**
- Easy to understand and interpret.
- Can handle both numerical and categorical data.
- Require minimal data preprocessing.
- Suitable for both classification and regression.
- Captures non-linear patterns
- Non-parametric, so no assumptions of linearity required

**Cons:**
- Prone to overfitting.
- Sensitive to noisy data
- Difficulty with imbalanced data
