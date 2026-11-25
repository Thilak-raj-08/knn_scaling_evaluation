import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


data = pd.read_csv("knn_k_customers.csv")

print("Loaded dataset shape:", data.shape)

print("\nFirst five customers:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nMissing values:")
print(data.isna().sum())

print("\nDuplicate rows:", data.duplicated().sum())

print("\nPurchase counts:")
print(data["Purchased"].value_counts())

features = [
    "Age",
    "Monthly_Spending",
    "Website_Visits",
    "Time_on_Website",
    "Previous_Purchases"
]

X = data[features]
y = data["Purchased"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

k = 7

unscaled_model = KNeighborsClassifier(
    n_neighbors=k,
    metric="euclidean"
)

scaled_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    ))
])

unscaled_model.fit(X_train, y_train)
scaled_model.fit(X_train, y_train)

unscaled_predictions = unscaled_model.predict(X_test)
scaled_predictions = scaled_model.predict(X_test)

print("\nModels trained successfully")

def evaluate_model(model_name, actual, predicted):
    print(f"\n{model_name} Evaluation")

    print("Accuracy:",
          round(accuracy_score(actual, predicted) * 100, 2), "%")

    print("Precision:",
          round(precision_score(actual, predicted, zero_division=0), 3))

    print("Recall:",
          round(recall_score(actual, predicted, zero_division=0), 3))

    print("F1-score:",
          round(f1_score(actual, predicted, zero_division=0), 3))

    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted))

evaluate_model(
    "Without Scaling",
    y_test,
    unscaled_predictions
)

evaluate_model(
    "With StandardScaler",
    y_test,
    scaled_predictions
)

k_values = [1, 3, 5, 7, 9, 11]
cv_results = []

for k_value in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=k_value,
            metric="euclidean"
        ))
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_results.append({
        "K Value": k_value,
        "Mean CV Accuracy (%)": round(scores.mean() * 100, 2)
    })

cv_table = pd.DataFrame(cv_results)

print("\nCross-validation results:")
print(cv_table.to_string(index=False))

best_cv_row = cv_table.loc[
    cv_table["Mean CV Accuracy (%)"].idxmax()
]

best_k = int(best_cv_row["K Value"])

print("\nBest K from cross-validation:", best_k)
print(
    "Best mean CV accuracy:",
    best_cv_row["Mean CV Accuracy (%)"],
    "%"
)

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean"
    ))
])

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

print("\nFinal selected model evaluation")
print("Selected K:", best_k)
print(
    "Test accuracy:",
    round(accuracy_score(y_test, final_predictions) * 100, 2),
    "%"
)

print("\nClassification report:")
print(classification_report(
    y_test,
    final_predictions,
    target_names=["No", "Yes"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, final_predictions))