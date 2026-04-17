import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# 1. Generate a non-linear dataset for binary classification
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# 2. Split the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Test Decision Tree for different criteria and tree depths
criteria = ["gini", "entropy"]
depths = [1, 2, 3, 4, 5, 10, 15, None]

best_tree_accuracy = 0
best_tree_params = None

print("\nDecision Tree results:")

for criterion in criteria:
    print(f"\nCriterion: {criterion}")

    for depth in depths:
        tree_model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=depth,
            random_state=42
        )

        tree_model.fit(X_train, y_train)
        y_pred = tree_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"max_depth={depth}, accuracy={accuracy:.4f}")

        if accuracy > best_tree_accuracy:
            best_tree_accuracy = accuracy
            best_tree_params = (criterion, depth)

print("\nBest Decision Tree:")
print(f"criterion={best_tree_params[0]}, max_depth={best_tree_params[1]}")
print(f"accuracy={best_tree_accuracy:.4f}")

# 4. Test Random Forest for different numbers of decision trees
n_estimators_values = [10, 50, 100, 200, 400, 800]

best_forest_accuracy = 0
best_forest_n = None

print("\nRandom Forest results:")

for n_estimators in n_estimators_values:
    forest_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    forest_model.fit(X_train, y_train)
    y_pred = forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"n_estimators={n_estimators}, accuracy={accuracy:.4f}")

    if accuracy > best_forest_accuracy:
        best_forest_accuracy = accuracy
        best_forest_n = n_estimators

print("\nBest Random Forest:")
print(f"n_estimators={best_forest_n}")
print(f"accuracy={best_forest_accuracy:.4f}")

# Scale features for models sensitive to feature magnitude
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Scale data and train Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_scaled, y_train)

y_pred_logreg = logreg_model.predict(X_test_scaled)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)

print("\nLogistic Regression result:")
print(f"accuracy={logreg_accuracy:.4f}")

# 5. Train SVM with RBF kernel
svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("\nSVM result:")
print(f"accuracy={svm_accuracy:.4f}")

# 6. Combine selected classifiers into a VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=best_forest_n, random_state=42))
    ],
    voting="soft"
)

voting_model.fit(X_train_scaled, y_train)
y_pred_voting = voting_model.predict(X_test_scaled)
voting_accuracy = accuracy_score(y_test, y_pred_voting)

# 7. Compare final results of all tested classifiers
results = {
    "Decision Tree": best_tree_accuracy,
    "Random Forest": best_forest_accuracy,
    "Logistic Regression": logreg_accuracy,
    "SVM": svm_accuracy,
    "Voting Classifier": voting_accuracy
}

print("\nFinal comparison of classifiers:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

best_model_name = max(results, key=results.get)
best_model_accuracy = results[best_model_name]

print("\nBest overall model:")
print(f"{best_model_name} with accuracy = {best_model_accuracy:.4f}")

print("\nVoting Classifier result:")
print(f"accuracy={voting_accuracy:.4f}")

# Visualize the decision boundary of the best model (SVM)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

Z = svm_model.predict(grid_scaled)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=10, cmap="coolwarm", edgecolors="none")

plt.title("Decision boundary of the best model (SVM)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
