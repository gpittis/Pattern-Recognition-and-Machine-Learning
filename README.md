# Pattern Recognition & Machine Learning - Assignment 2024-25

## Description
This assignment is part of the **Pattern Recognition & Machine Learning** course and consists of four main sections, each focusing on different **Classification Algorithms**.

## Part A: Maximum Likelihood Classifier
This part involves the development of a **Maximum Likelihood (ML) classifier** to recognize stress in users of a video game, based on data derived from button pressure patterns. The goal is to evaluate the reliability of the variable $x$. This evaluation is based on data from 12 users, 7 of whom did not feel stressed and 5 felt stressed.

### Implementation Steps:
- Estimation of the parameters $\hat{\theta}_1$ and $\hat{\theta}_2$ using the ML classifier for both classes:
  - For class ${\omega}_1$, the data is: ${D}_1$ = $[2.8,−0.4,−0.8,2.3,−0.3,3.6,4.1]$
  - For class ${\omega}_2$, the data is: ${D}_2$ = $[−4.5,−3.4,−3.1,−3.0,−2.3]$
- Plotting of $\log( p({D}_1 | \theta))$ and $\log( p({D}_2 | \theta))$ as functions of $\theta$
- Application of the discriminant function :
  - $g(x) = \log P(x | \hat{\theta}_1) - \log P(x | \hat{\theta}_2) + \log P(\omega_1) - \log P(\omega_2)$
- And Classification of the two sets of values.
- Plotting of $g(x)$ alongside the samples
  - The decision boundary is : $g(x) = 0$
- Evaluation of the classification performance
  - 1 sample is misclassified

## Part B: Bayesian Classifier
Part B is an extension of Part A. In this part, we implement a **Bayesian classifier** to estimate the unknown parameter $\theta$.

### Implementation Steps:
- Plotting of the posterior probability densities $p(\theta|D_1)$ and $p(\theta|D_2)$.
- Plotting of the prior probability density $p(\theta)$
- Observation of the relationship between the posterior probabilities and the prior $p(\theta)$
- Application of the discriminant function :
  - $h(x)$ = $\log P(x | D_1)$ - $\log P(x | D_2)$ + $\log P(\omega_1)$ - $\log P(\omega_2)$
- And Classification of the two sets of values.
- Plotting of $h(x)$ alongside the samples
  - The decision boundary is : $h(x) = 0$
- Evaluation of the classification performance
  - All samples were classified correctly

## Part C: Decision Tree Vs Random Forest
Use of the **Iris dataset** from the **sklearn library** and application of two classification algorithms:
- **Decision Tree**
- **Random Forest**
  
### Section 1: Decision Tree
- Isolation of the first two features of the dataset
- Use of the Decision Tree classifier
- Training on 50% of the data
- Evaluation on the remaining 50% of the data
- Finding the best classification accuracy
- Finding the optimal tree depth
- Plotting the decision boundaries of the classifier for the best result 

### Section 2: Random Forest
- Creation of a Random Forest classifier with 100 trees using the Bootstrap technique
- 50% of the samples from the previous training set are denoted as set A
- Creation of 100 new training sets (one for each tree), with each set using  $\gamma$ = 50% of set A
- Evaluation of the algorithm using the set classified in the previous section
- Consistency of the maximum depth across all trees
- Finding the best classification accuracy
- Finding the optimal tree depth
- Evaluation of the effect of $\gamma$ on the algorithm's performance.

## Part D: Classification Algorithm Development

- **Training Set**:  
  - `datasetTV.csv` 
  - 8743 samples with 224 features per sample, followed by a label (1,...,5) in the last column.
 
- **Test Set**:  
  - `datasetTest.csv`
  - 6955 samples without provided labels.
  
- **Implementation Steps**:  
  - Classification Algorithms Used in Part D:
    - k-Nearest Neighbors
    - Gaussian Naive Bayes
    - Multinomial Logistic Regression
    - Decision Trees
    - Random Forest
    - MLP (Multilayer Perceptron)
    - SVM (Support Vector Machine)
  - Training of classification algorithms on the training set
  - Determination of the best model for each classification algorithm
  - Finding the overall best model
  - Application of PCA (Principal Component Analysis) method to the overall best model
  - Application of the final, trained model to the test set.
  - Extraction of a vector named labels25.npy, representing the labels
  
## Collaborator
- [Anastasis Gourdomichalis](https://github.com/anasgourd)  
