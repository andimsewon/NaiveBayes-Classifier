# Naïve Bayes Classifier (Python)

A clean and lightweight implementation of the **Naïve Bayes classification algorithm** written in pure Python.
This repository was created for the *2024 Data Mining course* at Jeonbuk National University.

---

## Overview

This project builds a Naïve Bayes classifier without external libraries. The program:

1. Reads CSV data from training and test files
2. Calculates prior and conditional probabilities
3. Applies Laplace smoothing for unseen feature values
4. Predicts labels using the Maximum A Posteriori (MAP) rule
5. Prints predictions with probability scores

---

## How to Run

```bash
python 202219364_ML_hw2.py --train playtennis.csv --test playtennis_test.csv
```

**Example Output:**

```
Yes (0.85327)
No (0.14673)
```

---

## Dataset

* `playtennis.csv`: Original dataset from the lecture
* `smart_health_train.csv`, `smart_health_test.csv`: Custom dataset to verify model generalization

---

## Algorithm Details

The model uses **Bayes’ theorem** to compute posterior probabilities:

[
P(C|X) = \frac{P(X|C) * P(C)}{P(X)}
]

It assumes **conditional independence** between features and employs **Laplace smoothing** to prevent zero probabilities.

**Key steps:**

* Calculate prior probabilities for each class
* Count feature frequencies for each class
* Compute conditional probabilities
* Normalize and select the label with the highest posterior probability

---

## Comparison with Other Algorithms

| Algorithm     | Strengths                                         | Weaknesses                       |
| ------------- | ------------------------------------------------- | -------------------------------- |
| Naïve Bayes   | Fast, simple, effective with small or sparse data | Assumes feature independence     |
| Decision Tree | Easy to interpret, handles nonlinear data         | Prone to overfitting             |
| KNN           | Flexible, no training phase                       | High memory and computation cost |

---

## Author

**Kim Sewon (202219364)**
Department of Computer Engineering, Jeonbuk National University
