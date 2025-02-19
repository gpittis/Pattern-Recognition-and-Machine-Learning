# Pattern Recognition & Machine Learning - Assignment 2024-25

## Description
This assignment is part of the **Pattern Recognition & Machine Learning** course and consists of four main sections, each focusing on different classification and data analysis techniques.

## Part A: Maximum Likelihood Classifier
In this section, a **Maximum Likelihood (ML) classifier** is implemented based on frequency and key press force data from users during a game.

### Implementation Steps:
- Estimate parameters using the ML method.
- Visualize log-likelihood functions as a function of $\theta$.
- Implement the decision function and classify the data.
- Analyze and visualize the results.

## Part B: Bayesian Parameter Estimation
Extending the previous approach by estimating the unknown parameter \theta$ using **Bayesian inference**.

### Implementation Steps:
- Compute the posterior distribution $**p(\theta|D)**$.
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

## Author
**[Your Name]**  
