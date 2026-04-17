# moon-classification-model-comparison

A small machine learning project built with **scikit-learn** to compare several classifiers on the `make_moons` dataset. The project evaluates how different models handle a **non-linear binary classification** problem and includes a visualization of the decision boundary of the best-performing model.

## Project goal

The goal of this project is to:

- generate a synthetic non-linear dataset using `make_moons`,
- split the data into training and test sets,
- compare the performance of several classifiers,
- combine selected models using `VotingClassifier`,
- evaluate the final results,
- visualize the decision boundary of the best model.

## Models used

The following models were tested:

- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `LogisticRegression`
- `SVC` (SVM with RBF kernel)
- `VotingClassifier`

## Dataset

The dataset was generated with:

```python
make_moons(n_samples=10000, noise=0.4, random_state=42)
```

This creates a binary classification problem with overlapping, non-linearly separable classes.

## Workflow

1. Generate the dataset with `make_moons`.
2. Split the data into training and test subsets using `train_test_split`.
3. Test `DecisionTreeClassifier` with:
   - `gini`
   - `entropy`
   - different values of `max_depth`
4. Test `RandomForestClassifier` with different numbers of trees.
5. Train and evaluate:
   - `LogisticRegression`
   - `SVM` with RBF kernel
6. Combine `SVM`, `LogisticRegression`, and `RandomForestClassifier` using `VotingClassifier`.
7. Compare final accuracies and identify the best model.
8. Visualize the decision boundary of the best classifier.

## Results

Final comparison of classifiers:

- **Decision Tree:** `0.8555`
- **Random Forest:** `0.8470`
- **Logistic Regression:** `0.8380`
- **SVM:** `0.8640`
- **Voting Classifier:** `0.8600`

### Best model

The best-performing model was **SVM with RBF kernel**, which achieved an accuracy of **0.8640** on the test set.

## Key observations

- A shallow decision tree performed better than a deep one, which suggests overfitting for larger depths.
- Random Forest improved as the number of trees increased, but it did not outperform SVM.
- Logistic Regression achieved the weakest result, which is expected for a non-linear problem.
- SVM handled the curved class boundary best and achieved the highest accuracy.
- VotingClassifier improved over most single models, but it still did not outperform the best SVM model.

## Visualization

### Decision boundary of the best model (all data)

<!-- Replace the path below after adding your screenshot to the repository -->
![Decision boundary - all data](images/decision-boundary-all-data.png)

### Decision boundary of the best model (test data)

<!-- Replace the path below after adding your screenshot to the repository -->
![Decision boundary - test data](images/decision-boundary-test-data.png)

## Repository structure

```text
moon-classification-model-comparison/
├── classifiers.py
├── README.md
└── images/
    ├── decision-boundary-all-data.png
    ├── decision-boundary-test-data.png
```

## How to run

1. Clone the repository.
2. Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

3. Run the script:

```bash
python classifiers.py
```

## Notes

- `StandardScaler` was used before training `LogisticRegression` and `SVM`.
- The final visualization presents the decision boundary of the best-performing model.
- The project focuses on comparing classifiers on a synthetic non-linear dataset rather than optimizing hyperparameters extensively.

## Possible future improvements

- add confusion matrices for each classifier,
- compare training and test accuracy to analyze overfitting more explicitly,
- test additional SVM hyperparameters such as `C` and `gamma`,
- save plots automatically to files.
