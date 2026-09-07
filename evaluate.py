import pandas as pd
from sentiment import evaluate_model


df = pd.read_csv(
    "data/Clothing_Reviews.csv"
)


metrics = evaluate_model(df)


print("\nMODEL PERFORMANCE")
print("-----------------")

print(
    f"Best Configuration : "
    f"{metrics['Best Configuration']}"
)

print(
    f"Validation Accuracy: "
    f"{metrics['Validation Accuracy']:.4f}"
)

print(
    f"Test Accuracy      : "
    f"{metrics['Accuracy']:.4f}"
)

print(
    f"Weighted Precision : "
    f"{metrics['Precision']:.4f}"
)

print(
    f"Weighted Recall    : "
    f"{metrics['Recall']:.4f}"
)

print(
    f"Weighted F1 Score  : "
    f"{metrics['F1 Score']:.4f}"
)

print(
    f"Macro Precision    : "
    f"{metrics['Macro Precision']:.4f}"
)

print(
    f"Macro Recall       : "
    f"{metrics['Macro Recall']:.4f}"
)

print(
    f"Macro F1 Score     : "
    f"{metrics['Macro F1 Score']:.4f}"
)


print("\nConfusion Matrix:")
print(
    metrics["Confusion Matrix"]
)