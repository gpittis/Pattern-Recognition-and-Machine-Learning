# Pattern Recognition & Machine Learning - Assignment 2024-25

## Description
This assignment is part of the **Pattern Recognition & Machine Learning** course and consists of four main sections, each focusing on different **Classification Algorithms**.

## Part A: Maximum Likelihood Classifier
This part involves the development of a **Maximum Likelihood (ML) classifier** to recognize stress in users of a video game, based on data derived from button pressure patterns. The goal is to evaluate the reliability of the variable $x$. This evaluation is based on data from 12 users, 7 of whom did not feel stressed and 5 felt stressed.

### Implementation Steps:
- Estimate the parameters $\hat{\theta}_1$ and $\hat{\theta}_2$ using the ML classifier for both classes:
  - For class ${\omega}_1$, the data is: ${D}_1$ = $[2.8,−0.4,−0.8,2.3,−0.3,3.6,4.1]$
  - For class ${\omega}_2$, the data is: ${D}_2$ = $[−4.5,−3.4,−3.1,−3.0,−2.3]$
- Plot $\log( p({D}_1 | \theta))$ and $\log( p({D}_2 | \theta))$ as functions of $\theta$
- Use the discriminant function :
  - $g(x) = \log P(x | \hat{\theta}_1) - \log P(x | \hat{\theta}_2) + \log P(\omega_1) - \log P(\omega_2)$
- And classify the two sets of values.
- Plot $g(x)$ alongside the samples
  - The decision boundary is : $g(x) = 0$
- Evaluate the classification performance
  - 1 sample is misclassified

## Part B: Bayesian Parameter Estimation
Extending the previous approach by estimating the unknown parameter $\theta$ using **Bayesian inference**.

### Implementation Steps:
- Compute the posterior distribution $p(\theta|D)$.
- Visualize the posterior densities $p(\theta|D_1)$ and $p(\theta|D_2)$.
- Implement a **Bayesian classifier** using the decision function:

 $h(x)$ = $\log P(x | D_1)$ - $\log P(x | D_2)$ + $\log P(\omega_1)$ - $\log P(\omega_2)$

- Compare **ML** and **Bayesian** methods.

## Part C: Iris Classification
Using the **Iris dataset** from the `sklearn` library and applying two classification methods.

### Section 1: Decision Tree
- Use the `DecisionTreeClassifier`.
- Train on **50%** of the data, evaluate on the remaining **50%**.
- Report classification accuracy and determine the **optimal depth**.
- Visualize **decision boundaries**.

### Section 2: Random Forest
- Create a **Random Forest classifier** with **100 trees** using **bootstrap sampling**.
- Compare with the simple **Decision Tree**.
- Analyze the impact of **$\gamma$** (sample percentage) on performance.

## Part D: Large Dataset Classification
Using the dataset **TV.csv** to train a classifier with **8743 samples** and **224 features**.

### Implementation Steps:
- **Feature selection** and **preprocessing**.
- Train a classifier (e.g., **SVM, Random Forest, Neural Networks**).
- Evaluate the model on **datasetTest.csv**.
- Export predictions as a **numpy array**.

## Collaborator
- [Anastasis Gourdomichalis](https://github.com/anasgourd)  
